from os import getenv
from ywmem import redis_ops, working_memory
from ywmem.rdf import CONV
from yltmem.longterm_memory import LongTermMemory
import json
import datetime
import uuid
from time import sleep
import hashlib
from typing import List, Dict

from api.exceptions import SessionError

import logging

logger = logging.getLogger(__name__)

USE_LONGTERM = getenv("USE_LONGTERM", False)

KEY_ACTIVE_TIME = datetime.timedelta(
    hours=1
)  # within the hour, the conversation will be picked up

KEY_POTENTIAL_TIME = datetime.timedelta(
    days=30
)  # within a month, the bot will ask if the user wants to continue


class Session:
    """
    Managing Sessions by identifying users by channel of conversation & custom identifiers.
    Session keys are saved in redis under a "user_key" (source_user_identifiers).
    If a generated user_key has keys in redis, then the last modified time is checked.
    If the key is still active, confirmation to continue conversation is asked.
    If the key is semi-active, the conversation key is marked as potential.
    If there are no matching keys for the user_key, then the user_identifiers are checked.
    This allows channel_switching, as only one user identifier needs to match, and not the channel.
    """

    def __init__(
        self,
        channel: str,
        user_identifiers: Dict[str, str],
        communication_type="chat",
        session_id=None,
        pickled=True,
    ):
        self.communication_type = communication_type
        self.channel = channel
        self.user_identifiers = user_identifiers
        self.id = None
        self.closed = False
        self.needs_confirmation = False
        self.is_potential = False
        self.needs_lexicon = False
        self.is_new = False
        self.pickled = pickled
        self.language = None
        self.language_to = None
        self.device = {"screen": True, "location": False, "browser": False}
        self.wm = None
        self.is_busy = False

        for id in user_identifiers:
            uid = user_identifiers[id]
            if id != "phone_number" and id != "_phone_number":
                self.user_identifiers[id] = hashlib.sha256(uid.encode()).hexdigest()

        if session_id:
            if not self.load_by_id(session_id):
                err = SessionError("Failed to load by ID, not found.", self)
                raise err

        else:
            self.user_key = f"keys_{channel}"
            closed_keys = []  # Filled with closed keys in following function
            is_loaded = self._load_from_user_key(closed_keys)
            if is_loaded:
                return
            is_loaded = self._load_from_user_ids(closed_keys)
            if is_loaded:
                return

            # nothing returned yet, so need to generate an id
            self.generate_id()
            self.update_in_redis()

    def __enter__(self):
        """
        Upon entering 'with Session() as session'
        Checks if the session is free to use in Redis
        i.e. not being used by another API call and sets the session as busy in Redis

        :return: the session instance
        :rtype: Session
        """
        self._wait_until_session_is_free()
        self._mark_session_busy()
        return self

    def __exit__(self, *args):
        """
        Upon exiting the 'with Session() as session'
        Marks the session as free in Redis
        """
        self._mark_session_free()

    def load_by_id(self, key_id: str) -> bool:
        """
        Loops through the keys in Redis to see which key holds a certain conversation.
        The conversation that should be loaded has a certain ID

        :param key_id: The ID of the conversation
        :type key_id: str
        :return: Returns False if there is no conversation with this ID
        :rtype: bool
        """
        all_keys = [key for key in redis_ops.get_all_keys() if "keys_" in str(key)]
        for user_key in all_keys:
            keys = json.loads(redis_ops.get(user_key))
            for key in keys:
                if key["id"] == key_id:
                    self.user_key = user_key
                    self.load(key)
                    return True
        return False

    def load(self, key_obj: dict):
        """
        Loads the information of a key from Redis into class attributes.

        :param key_obj: Information from Redis
        :type key_obj: dict
        """
        if not self.user_key:
            self.load_by_id(key_obj["id"])  # this function provides user_key

        self.id = key_obj["id"]
        self.language = key_obj["language"]
        self.last_modified = key_obj["last_modified"]
        self.closed = key_obj["closed"]
        self.user_identifiers = key_obj["user_identifiers"]
        self.uri = CONV[self.id]
        self.pickled = key_obj.get("pickled", False)
        self.is_busy = key_obj.get("is_busy")
        self.channel = key_obj.get("channel")

    def serialize(self) -> dict:
        """
        Create dict from Class information for ID
        """
        self.last_modified = datetime.datetime.isoformat(datetime.datetime.now())
        return {
            "id": self.id,
            "language": self.language,
            "last_modified": self.last_modified,
            "user_identifiers": self.user_identifiers,
            "closed": self.closed,
            "pickled": self.pickled,
            "is_busy": self.is_busy,
            "channel": self.channel,
        }

    def update_in_redis(self):
        """
        Update ID information in redis under user keys.
        """
        key_obj = self.serialize()
        if redis_ops.key_exists(self.user_key):
            found = False
            full_key_list = json.loads(redis_ops.get(self.user_key))
            for index, key in enumerate(full_key_list):
                if self.id == key["id"]:
                    full_key_list[index] = key_obj
                    found = True

            if not found:
                full_key_list.append(key_obj)

            redis_ops.persist_to_redis(self.user_key, json.dumps(full_key_list))
        else:
            redis_ops.persist_to_redis(self.user_key, json.dumps([key_obj]))

    def generate_id(self):
        """
        Generate ID for a WM.
        """
        self.id = uuid.uuid4().hex
        self.uri = CONV[self.id]
        self.is_new = True
        self.pickled = (
            self.pickled
        )  # from now on we pickle them, so set to true by default

    def close(self):
        """
        Close a session to make sure the WM is never used again.
        """
        if self.closed:
            return

        self.closed = True
        self.update_in_redis()

    def destroy(self):
        """
        Clear a complete WM, along with its ID in redis
        """
        self.close()
        # redis_ops.delete(self.id)
        logger.debug(f"Destroyed session with ID {self.id}")

    def _wait_until_session_is_free(self):
        """
        Checks if the session is marked in Redis as "busy"
        By loading the redis key entry and checking if is_busy is set to True.
        If not, waits for a second and tries again
        """
        while self.is_busy:
            logger.debug("Waiting until session is free.")
            sleep(1)
            self.load_by_id(self.id)

    def _mark_session_busy(self):
        """
        Marks a session as busy in Redis by setting "is_busy" to True.
        """
        self.is_busy = True
        self.update_in_redis()

    def _mark_session_free(self):
        """
        Marks a session as free in Redis by setting "is_busy" to False.
        """
        self.is_busy = False
        self.update_in_redis()

    def load_wm(self, only_from_redis: bool = False) -> working_memory.WorkingMemory:
        """Load WM by ID from Redis or LTM. If nothing is found, new WM is created.
        If self.id is known, it will check Redis and LTM for known WM's with this ID.
        If there is no such WM, it will create a WM with this ID and persist it in Redis.

        :param only_from_redis: If set to True will skip checking LTM for a conversation with this ID, defaults to False
        :type only_from_redis: bool, optional
        :raises err: if session has no ID, it will raise a SessionError
        :return: the loaded or created WM
        :rtype: working_memory.WorkingMemory
        """
        if not self.id:
            err = SessionError("No ID found", self)
            raise err

        if redis_ops.key_exists(self.id):
            self.wm = redis_ops.retrieve(self.id, pickled=self.pickled)
            self.wm.set_spacy_model_endpoint(getenv("NLP_HOSTNAME"), getenv("NLP_PORT"))
            return self.wm

        # there is no redis entry for the id in the session
        elif not self.is_new and not only_from_redis and USE_LONGTERM:
            # check the ltm if there is a graph with this id
            ltm = LongTermMemory()
            graph = ltm.get_graph(self.uri)
            self.wm = working_memory.WorkingMemory(self.id, pickled=self.pickled)
            self.wm.set_spacy_model_endpoint(getenv("NLP_HOSTNAME"), getenv("NLP_PORT"))
            try:
                self.wm.from_graph(graph)
                logger.debug("Loaded graph from LTM")
            except Exception:
                logger.error("Received empty graph from LTM")
                self.is_new = True

            self.needs_lexicon = True
            return self.wm
        else:
            # nothing in redis or ltm, create a new one
            self.wm = working_memory.WorkingMemory(
                self.id, pickled=self.pickled, channel=self.channel
            )
            self.wm.set_spacy_model_endpoint(getenv("NLP_HOSTNAME"), getenv("NLP_PORT"))
            self.needs_lexicon = True
            return self.wm

    def persist_to_LTM(self):
        """
        If session has ID and WM in Redi, move to LTM.
        """
        # persist WM to LTM
        if not self.id:
            err = SessionError("No ID found for this Session", self)
            raise err
        if not self.wm and not self.load_wm(only_from_redis=True):
            return False
        ltm = LongTermMemory()
        if ltm.set_graph(self.wm.graph, self.uri):
            logger.debug("Uploaded WM to LTM")
            if redis_ops.delete(self.id):
                logger.debug("Removed WM from Redis")
                return True
            else:
                err = SessionError("Could not remove WM from Redis", self)
                raise err
        else:
            err = SessionError("Could not upload WM to LTM", self)
            raise err

    def age(self):
        last_mod = datetime.datetime.fromisoformat(self.last_modified)
        return datetime.datetime.now() - last_mod

    def _load_from_user_key(self, closed_keys: List) -> bool:
        """
        Returns if user was loaded. The id's of keys found with argument `closed` True
        are added to `closed_keys`.
        """
        # generate user_key, to get all the possible conversation ids
        for type in self.user_identifiers:
            self.user_key += f"_{self.user_identifiers[type]}"
        if redis_ops.key_exists(self.user_key):
            keys = json.loads(redis_ops.get(self.user_key))
            key_ages = {}
            for key in keys:
                key_last_modified = datetime.datetime.fromisoformat(
                    key["last_modified"]
                )
                key_age = datetime.datetime.now() - key_last_modified
                if key["closed"]:
                    closed_keys.append(key["id"])
                    continue
                if key_age < KEY_ACTIVE_TIME:
                    # key is still active
                    self.load(key)
                    return True
                # too old
                else:
                    key_ages[key["id"]] = key_age.total_seconds()

            # there are keys found, but none of them are young enough
            if len(key_ages) > 0:
                newest_key = sorted(key_ages.items(), key=lambda x: x[1])[0]
                newest_key_id = newest_key[0]
                newest_key_age = newest_key[1]
                if newest_key_age < KEY_POTENTIAL_TIME.total_seconds():
                    # ask if conversation needs to continue
                    self.load([key for key in keys if key["id"] == newest_key_id][0])
                    self.is_potential = True
                    return True
        return False

    def _load_from_user_ids(self, closed_keys: List) -> bool:
        """
        Finds the user session by looking in Redis if one of the user IDs match.
        Loads the user session information into the class instance.
        Returns if user was loaded.

        :param closed_keys: a list of keys to sessions that are no longer active.
        """
        all_keys = [key for key in redis_ops.get_all_keys() if "keys_" in str(key)]
        for user_key in all_keys:
            keys = json.loads(redis_ops.get(user_key))
            for key in keys:
                if key["closed"] or key["id"] in closed_keys:
                    continue
                key_last_modified = datetime.datetime.fromisoformat(
                    key["last_modified"]
                )
                key_age = datetime.datetime.now() - key_last_modified
                for identifier in self.user_identifiers:
                    if (
                        identifier in key["user_identifiers"]
                        and self.user_identifiers[identifier]
                        == key["user_identifiers"][identifier]
                        and key_age < KEY_POTENTIAL_TIME
                        and _identifier_is_binding(identifier)
                    ):
                        # we found a match by identifier (tel_number, email)
                        self.load(key)
                        self.is_potential = True
                        return True
        return False


def _identifier_is_binding(identifier: str) -> bool:
    """Checks whether the identifier is binding, i.e. directly identifies the user without other identifiers.
    This functionality allows us to create identifiers that are not binding by adding an _ before it.
    This means the identifier will be added to the graph and returned in the endpoints, but won't be used when checking what sessions are active.

    :param identifier: the identifier in question
    :type identifier: str
    :return: whether the identifier is binding
    :rtype: bool
    """
    return identifier[0] != "_"


def _get_session_wm(session_id: str) -> working_memory.WorkingMemory:
    """
    Helper to get an instance of the working memory of a session

    :return: None or a working memory
    """
    # See if the session keys still exists in the working memory

    session = _get_session_by_id(session_id)
    return session.wm


def _get_session_by_id(session_id: str):
    session = Session(
        channel=None,
        user_identifiers={},
        session_id=session_id,
    )
    session.load_wm()
    return session


def _get_session_data(session_id: str) -> Session:
    """
    Get the information about a session

    :return: None or a dict with information about the session
    """
    session = _get_session_by_id(session_id)

    if not session or not session.wm:
        logger.debug(f"session for {session_id} not found")
        return None

    # TODO figure out what the "proper" channel is
    turns = [
        turn.uri.n3(session.wm.graph.namespace_manager) for turn in session.wm.turns
    ]

    return {"id": session_id, "channel": session.channel, "turns": turns}
