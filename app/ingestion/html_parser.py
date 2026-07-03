from bs4 import Tag, BeautifulSoup
from pydantic.dataclasses import dataclass


@dataclass
class RawSection:
    section: str
    text: str
    tags: list[str]

_NOISE = ["script", "style", "nav", "footer", "header", "aside", "button"]

def _clean(node: Tag) -> str:
    for bad in node.find_all(_NOISE):
        bad.decompose()
    return " ".join(node.get_text(separator=" ", strip=True).split())

def parse_sheet(html: str, *, subject_hint="", sheet_hint=""):
    soup = BeautifulSoup(html, "lxml")

    title_el = soup.select_one("h1")
    sheet = (title_el.get_text(strip=True) if title_el else "") or sheet_hint
    subject = subject_hint

    sections: list[RawSection] = []
    nodes = soup.select("section")
    if nodes:
        for node in nodes:
            head = node.find(["h2", "h3", "h4"])
            name = head.get_text(strip=True) if head else ""
            tags = [t.get_text(strip=True) for t in node.select(".tag")]
            body = _clean(node)
            if body:
                sections.append(RawSection(name, body, tags))
    else:
        cur = RawSection(sheet, "", [])
        for element in (soup.body or soup).descendants:
            if isinstance(element, Tag) and element.name in ("h2", "h3"):
                if cur.text.strip():
                    sections.append(cur)
                cur = RawSection(element.get_text(strip=True), "", [])
            elif isinstance(element, Tag) and element.name in ("p", "li", "pre", "code"):
                cur.text += " " + element.get_text(" ", strip=True)
        if cur.text.strip():
            sections.append(cur)
    return subject, sheet, sections