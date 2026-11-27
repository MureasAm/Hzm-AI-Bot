"""Ground forms of address and conversational roles in authored private text."""
import re

_TRANSPORT_BLOCK = re.compile(r"\[(?:引用|转发|图片|截图|语音描述)[^\]]*\]", re.S)
_REPORTED_QUOTE = re.compile(
    r"(?:说(?:道|过)?|问(?:道|过)?|写(?:道|过)?|回复|转述|原话是)"
    r"\s*[:：]?\s*[「『“\"][^」』”\"]*[」』”\"]"
)
_REPORTED_CLAUSE = re.compile(r"(?:他说|她说|对方说|别人说|朋友说|同学说|同事说)[：:，,]\s*[^。！？!?\n]+")
_CLAUSE = re.compile(r"[^，,。！？!?\n；;]+[。！？!?]?")
_PREFACE = re.compile(r"^(?:嗯+|哦+|呃+|那个|对了|总之|其实|好吧|所以|那(?=我|你|姐姐|小满|灰泽满))\s*")
_SHE = r"(?:你|灰泽满|小满|姐姐|hzm)"
_TOKEN = r"[\w.·-]{1,24}?"
_NAME_QUOTES = "「」『』“”‘’\"'"
_OWN_NAME = re.compile(
    r"^(?:(?P<intro>我(?:的)?(?:本名|真名|网名|昵称|外号|名字|曾用名|艺名)(?:是|叫|为)|"
    r"我(?:叫|是|自称))|你(?:平时|经常|常)?(?:叫|喊|称呼)我的(?:外号|昵称|名字)(?:是|叫|为)|"
    r"(?:(?:以后|今后|平时|平常|从今以后|这次|就|请|麻烦|你|能不能|可不可以|可以|能|"
    r"经常|常常|常|一直|只要|要是害羞|希望你|想让你)\s*)*(?:(?:叫|喊|称呼)我(?:为)?|管我叫))"
    r"\s*",
    re.I,
)
_CONTINUED_NAME = re.compile(r"^(?:本名|真名|网名|昵称|外号|名字|曾用名)(?:可以)?(?:是|叫|为)\s*")
_QUESTION_END = re.compile(r"[？?]$|[吗嘛么][。！!]*$")
_NAME_END = re.compile(r"\s*(?:就好|就行|即可|就可以(?:了)?|好了|就对了|也行|可以)?(?:吧|啦|啊|呢|嘛|吗|呀)?[。！？!?]*$")
_NAME_LIST_SEPARATOR = re.compile(r'、|/|／|或者|或|和|(?<=[」』”"])\s*(?=[「『“"])')
_HER_DEFINITION = re.compile(
    rf"^{_SHE}(?:指的)?就是\s*(?:我的|我家)?[{_NAME_QUOTES}]?(?P<after>{_TOKEN})[{_NAME_QUOTES}]?"
    rf"(?:啊|呀|啦|哦|呢)?[。！!]*$"
    rf"|^(?:我说的|我叫的|我家|我的)?[{_NAME_QUOTES}]?(?P<before>{_TOKEN})[{_NAME_QUOTES}]?"
    rf"\s*(?:指的)?就是\s*{_SHE}(?:啊|呀|啦|哦|呢)?[。！!]*$",
    re.I,
)
_NON_NAMES = {"我", "你", "她", "他", "它", "我们", "你们", "她们", "他们", "对方", "这个", "那个"}
_USER_TO_HER = re.compile(
    rf"^(?:{_SHE}[~～\s]*)?我(?:真的|好想|特别|有点|很|好|也|真|最|太)*"
    rf"(?P<verb>想念|喜欢|爱|担心|想){_SHE}(?=$|[啦呀啊呢嘛哦了！!。？?]|怎么办)",
    re.I,
)
_HER_TO_USER = re.compile(
    rf"^{_SHE}(?:到底|真的|还|也|有点)?"
    r"(?P<verb>喜不喜欢|想不想|爱不爱|喜欢|想念|想|爱|担心)我(?:呀|啊|嘛|吗|么|呢|不)?[。！？!?]*$",
    re.I,
)
_ATTITUDE_QUESTIONS = {"喜不喜欢", "想不想", "爱不爱"}
_ATTITUDES = {"喜不喜欢": "喜欢", "想不想": "想念", "爱不爱": "爱", "想": "想念"}


def author_text(text: str) -> str:
    """Exclude quoted speech and adapter-added details from attribution."""
    text = _TRANSPORT_BLOCK.sub("", text or "")
    return _REPORTED_CLAUSE.sub("", _REPORTED_QUOTE.sub("", text)).strip()


def _clauses(text: str):
    for match in _CLAUSE.finditer(author_text(text)):
        clause = _PREFACE.sub("", match[0].strip()).strip()
        if clause:
            yield clause


def _literal(text: str) -> str:
    return text.strip().strip(_NAME_QUOTES).strip().casefold()


def user_address_matches(text: str, value: str) -> bool:
    """A name must follow an explicit reference to the current author."""
    value = str(value or "").strip()
    if not value:
        return False
    names = {_literal(term) for term in _NAME_LIST_SEPARATOR.split(value)}
    observed = set()
    continuing = False
    for clause in _clauses(text):
        cue = _OWN_NAME.match(clause) or (_CONTINUED_NAME.match(clause) if continuing else None)
        continuing = bool(cue and not (cue.groupdict().get("intro") and _QUESTION_END.search(clause)))
        if not continuing:
            continue
        literal_body = clause[cue.end():].strip().rstrip("。！？!?")
        if _literal(literal_body) == _literal(value):
            return True
        body = _NAME_END.sub("", literal_body).strip()
        if _literal(body) == _literal(value):
            return True
        observed.update(_literal(term) for term in _NAME_LIST_SEPARATOR.split(body))
    return bool(names and "" not in names and names <= observed)


def her_alias_definition(text: str) -> str:
    """Read explicit current definitions such as ‘姐姐就是我的宝宝’."""
    definition = ""
    for clause in _clauses(text):
        if _QUESTION_END.search(clause):
            continue
        match = _HER_DEFINITION.fullmatch(clause)
        if match:
            term = match["after"] or match["before"]
            if term not in _NON_NAMES:
                definition = term
    return definition


def perspective_note(text: str) -> str:
    """Render a short fact about this turn when the direction is explicit."""
    facts = []
    alias = her_alias_definition(text)
    if alias:
        facts.append(f"用户本轮说「{alias}」指灰泽满，按本轮这个指代接话。")
    for clause in _clauses(text):
        outgoing = _USER_TO_HER.match(clause)
        incoming = _HER_TO_USER.fullmatch(clause)
        if outgoing and (not _QUESTION_END.search(clause) or "怎么办" in clause):
            verb = _ATTITUDES.get(outgoing["verb"], outgoing["verb"])
            facts.append(f"用户对灰泽满表达心意，用户{verb}灰泽满。")
        if incoming and (incoming["verb"] in _ATTITUDE_QUESTIONS or _QUESTION_END.search(clause)):
            verb = _ATTITUDES.get(incoming["verb"], incoming["verb"])
            facts.append(f"用户在问灰泽满是否{verb}用户。")
    if not facts:
        return ""
    return "【本轮说话方向】\n" + "\n".join(dict.fromkeys(facts)) + "\n用灰泽满的视角回应。"
