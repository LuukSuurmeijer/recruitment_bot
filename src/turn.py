import uuid
import json
from rdflib import Literal, Graph
from rdflib.namespace import XSD
from rdflib.term import URIRef
from datetime import datetime
from typing import Optional, List

from ywmem.utterance import Utterance, Sentence
from ywmem.rdf import Y, DF, CONV
import logging

logger = logging.getLogger(__name__)
get_last_utt_by_turn = """
    SELECT DISTINCT ?utt WHERE {
        ?turn conv:hasPart ?utt .
        FILTER NOT EXISTS {?someOtherUtt conv:succeedes ?utt .}
    }
"""


class Turn:
    def __init__(
        self,
        participant,
        time: Optional[datetime] = None,
        turn_uri: Optional[URIRef] = None,
        draft=False,
    ):
        """
        Args:
            ...
            turn_uri: URI reference to the turn in the graph. New URI is generated when
                set to None.
        """
        self.participant: URIRef = participant
        self.utterances: List[Sentence] = []
        self.matched_intent_name = None
        self.parameters = None
        self.time = time if time else datetime.now()
        self.uri = turn_uri if turn_uri else CONV[f"turn_{uuid.uuid4().hex}"]
        self.draft = draft

    def append_utterance(self, utterance: Utterance):
        self.utterances.append(utterance)
        if len(self.utterances) > 1:
            utterance.previous = self.utterances[-2]
        utterance.turn = self
        return self

    def set_matched_intent(self, intentName):
        self.matched_intent_name = intentName
        return self

    def set_parameters(self, parameters):
        self.parameters = parameters
        return self

    def write_utterance(self, graph, utterance, spacy_model):
        utterance.write(graph, spacy_model)  # write utterance to graph
        last_utt = next(
            (
                row.utt
                for row in graph.query(
                    get_last_utt_by_turn, initBindings={"turn": self.uri}
                )
            ),
            None,
        )
        graph.add(
            (self.uri, CONV.hasPart, utterance.uri)
        )  # write utterance as a part of turn
        if last_utt:
            graph.add(
                (utterance.uri, CONV.succeedes, last_utt)
            )  # if its not the first utterance, add succession

    def write(self, graph: Graph, turn_node, spacy_model, only_write_utt=None):
        last_utt = None
        utt: Utterance
        for utt in self.utterances:
            if not only_write_utt or utt == only_write_utt:
                utt_node = utt.uri
                if utt_node:
                    utt.write(graph, spacy_model)
                    graph.add((turn_node, CONV.hasPart, utt_node))
                    if last_utt:
                        graph.add((utt_node, CONV.succeedes, last_utt))
                    last_utt = utt_node
        if self.matched_intent_name and only_write_utt is None:
            graph.add(
                (
                    turn_node,
                    DF.matchesIntentName,
                    Literal(self.matched_intent_name, datatype=XSD.string),
                )
            )
        if self.parameters and only_write_utt is None:
            graph.add(
                (
                    turn_node,
                    DF.hasParameters,
                    Literal(json.dumps(self.parameters), datatype=XSD.string),
                )
            )
        if only_write_utt is None:
            graph.add(
                (
                    turn_node,
                    Y.timeStamp,
                    Literal(self.time.isoformat(), datatype=XSD.dateTime),
                )
            )
            graph.add((turn_node, CONV.isDraft, Literal(self.draft)))

    def reset_translation(self, wm):
        """Resets translation of all utterances in the turn

        :param wm: _description_
        :type wm: _type_
        """
        for utt in self.utterances:
            if utt.translated_from:
                self.utterances[self.utterances.index(utt)] = utt.translated_from
                utt.translated_from.potential_translation = utt
                utt.remove_links(wm)
                utt.write_links(wm)
                utt.translated_from.turn = self  # make sure the utt now has the turn
                utt.translated_from.remove_links(wm)
                utt.translated_from.write_links(wm)
        wm.persist()

    def serialize(self, wm):
        """
        Serializes the content of a specific turn to make it consumable in a flask jsonify method

        :param turn: An instance of a Turn from the working memory
        :param wm: The working memory itself
        :return: A dictionary with the information of the turn
        """
        utterances = [utterance.serialize(wm) for utterance in self.utterances]
        participant = wm._get_participant_by_uri(self.participant)

        return {
            "id": self.uri.n3(wm.graph.namespace_manager),
            "participant": participant,
            "utterances": utterances,
            "created": self.time.isoformat(),
            "is_draft": self.draft,
        }

    def remove_links(self, wm):
        wm.graph.remove((wm.conv, CONV.hasPart, self.uri))
        wm.graph.remove((self.uri, CONV.succeedes, None))
        wm.graph.remove((None, CONV.succeedes, self.uri))
