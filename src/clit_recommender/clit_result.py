from dataclasses import dataclass
from typing import List, Optional, Dict
from dataclasses_json import dataclass_json, LetterCase


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Logger:
    log_name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Mention:
    mention: str
    offset: int
    assignment: Optional[str]
    detection_confidence: float
    possible_assignments: List[str]
    original_mention: str
    original_without_stopwords: str
    logger: Logger


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Document:
    uri: Optional[str]
    text: str
    mentions: List[Mention]
    component_id: str
    pipeline_type: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class PipelineConfig:
    start_components: List[str]
    components: Dict[str, List[Dict[str, str]]]
    example_id: str
    end_components: List[str]
    display_name: str
    id: int
    connections: List[str]
    pipeline_config_type: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ExperimentTask:
    experiment_id: int
    task_id: int
    documents: List[List[Document]]
    pipeline_config: PipelineConfig
    pipeline_type: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ClitResult:
    experiment_tasks: List[ExperimentTask]
    experiment_id: int
