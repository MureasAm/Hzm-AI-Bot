"""Bounded interpretation of one turn, independent of retrieval candidates."""
from dataclasses import dataclass
from enum import Enum
import json


class TurnFocus(str, Enum):
    CHARACTER = 'character'
    USER = 'user'
    THIRD_PARTY = 'third_party'
    SHARED = 'shared'
    GENERAL = 'general'
    UNCLEAR = 'unclear'


@dataclass(frozen=True)
class GroundedParticipant:
    name: str
    evidence: str

    def to_dict(self) -> dict:
        return {'name': self.name, 'evidence': self.evidence}


@dataclass(frozen=True)
class TurnUnderstanding:
    focus: TurnFocus
    focus_text: str
    request: str
    exclusions: tuple[str, ...]
    premises: tuple[str, ...]
    third_parties: tuple[GroundedParticipant, ...]

    def to_dict(self) -> dict:
        return {
            'focus': self.focus.value, 'focus_text': self.focus_text, 'request': self.request,
            'exclusions': list(self.exclusions), 'premises': list(self.premises),
            'third_parties': [person.to_dict() for person in self.third_parties],
        }


def _short_text(value, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if 0 < len(text) <= limit else None


def _short_list(value) -> tuple[str, ...] | None:
    if not isinstance(value, list) or len(value) > 6:
        return None
    items = []
    for raw in value:
        item = _short_text(raw, 160)
        if item is None:
            return None
        if item not in items:
            items.append(item)
    return tuple(items)


def parse_understanding(value, raw_text: str, history: str) -> TurnUnderstanding | None:
    """Validate structure and literal participant provenance, without inferring facts."""
    if not isinstance(value, dict):
        return None
    try:
        focus = TurnFocus(value.get('focus'))
    except (TypeError, ValueError):
        return None
    focus_text = _short_text(value.get('focus_text'), 160)
    request = _short_text(value.get('request'), 320)
    exclusions = _short_list(value.get('exclusions'))
    premises = _short_list(value.get('premises'))
    people = value.get('third_parties')
    if (focus_text is None or request is None or exclusions is None or premises is None
            or not isinstance(people, list) or len(people) > 6):
        return None
    participants = []
    for person in people:
        if not isinstance(person, dict):
            return None
        name = _short_text(person.get('name'), 64)
        evidence = _short_text(person.get('evidence'), 240)
        if (name is None or evidence is None or name not in evidence
                or not (evidence in raw_text or evidence in history)):
            return None
        participant = GroundedParticipant(name, evidence)
        if participant not in participants:
            participants.append(participant)
    return TurnUnderstanding(focus, focus_text, request, exclusions, premises, tuple(participants))


def render_understanding(understanding: TurnUnderstanding | None) -> str:
    """Generation gets the same boundaries, without copying participant evidence."""
    if understanding is None:
        return ''
    labels = {
        TurnFocus.CHARACTER: '角色本人', TurnFocus.USER: '当前用户',
        TurnFocus.THIRD_PARTY: '第三人', TurnFocus.SHARED: '双方共同话题',
        TurnFocus.GENERAL: '通用任务', TurnFocus.UNCLEAR: '尚不明确',
    }
    fields = {
        '人物与焦点': labels[understanding.focus] + '：' + understanding.focus_text,
        '实际需求': understanding.request, '排除范围': list(understanding.exclusions),
        '待核实前提（不能当事实）': list(understanding.premises),
        '有原话依据的第三人': [person.name for person in understanding.third_parties],
    }
    return ('本轮理解参考，原话优先；待核实前提不能当事实，排除范围仅限制对应用途。\n'
            + json.dumps(fields, ensure_ascii=False))
