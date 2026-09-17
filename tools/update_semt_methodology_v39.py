from copy import deepcopy
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile


BASE = Path("/Users/mraffyzeidan/Downloads/SEMT/SEMT_updated_introduction.docx")
METHOD = Path("/private/tmp/journal-method-v39.docx")
OUTPUT = Path("/Users/mraffyzeidan/Downloads/SEMT/SEMT_full_journal_v39.docx")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XML_NS = "http://www.w3.org/XML/1998/namespace"
W = f"{{{W_NS}}}"
ET.register_namespace("w", W_NS)
ET.register_namespace("m", M_NS)
ET.register_namespace("r", R_NS)


def text(element):
    return "".join(node.text or "" for node in element.iter(W + "t")).strip()


def style(element):
    node = element.find("./" + W + "pPr/" + W + "pStyle")
    return node.get(W + "val") if node is not None else ""


def strip_bookmarks(element):
    for parent in element.iter():
        for child in list(parent):
            if child.tag in {W + "bookmarkStart", W + "bookmarkEnd"}:
                parent.remove(child)


def get_or_add(parent, tag, before=None):
    node = parent.find(tag)
    if node is None:
        node = ET.Element(tag)
        if before is None:
            parent.append(node)
        else:
            parent.insert(before, node)
    return node


def set_paragraph_layout(paragraph, first_line=True, centered=False):
    ppr = paragraph.find(W + "pPr")
    if ppr is None:
        ppr = ET.Element(W + "pPr")
        paragraph.insert(0, ppr)

    jc = get_or_add(ppr, W + "jc")
    jc.set(W + "val", "center" if centered else "both")

    ind = get_or_add(ppr, W + "ind")
    ind.attrib.pop(W + "hanging", None)
    if first_line:
        ind.set(W + "firstLine", "720")
    else:
        ind.attrib.pop(W + "firstLine", None)
        ind.set(W + "firstLine", "0")


def format_body_paragraph(paragraph):
    paragraph_style = style(paragraph)
    value = text(paragraph)
    if not value or paragraph_style.startswith("Heading"):
        return
    if paragraph.find(".//" + f"{{{M_NS}}}" + "oMath") is not None and not value:
        return
    is_caption = value.startswith(
        (
            "Table 1.", "Table 2.", "Table 3.",
            "Figure 1.", "Figure 2.", "Figure 3.", "Figure 4.",
        )
    )
    set_paragraph_layout(
        paragraph,
        first_line=not is_caption,
        centered=is_caption,
    )
    if is_caption:
        ppr = paragraph.find(W + "pPr")
        keep_next = get_or_add(ppr, W + "keepNext")
        keep_next.set(W + "val", "1")


def set_table_border(borders, name, value, size="8"):
    border = get_or_add(borders, W + name)
    border.set(W + "val", value)
    if value != "nil":
        border.set(W + "sz", size)
        border.set(W + "space", "0")
        border.set(W + "color", "000000")


def make_algorithm_title_row(template_row):
    row = ET.Element(W + "tr")
    trpr = template_row.find(W + "trPr")
    if trpr is not None:
        row.append(deepcopy(trpr))
    cell = ET.SubElement(row, W + "tc")
    tcpr = ET.SubElement(cell, W + "tcPr")
    span = ET.SubElement(tcpr, W + "gridSpan")
    span.set(W + "val", "2")
    paragraph = ET.SubElement(cell, W + "p")
    set_paragraph_layout(paragraph, first_line=False, centered=True)
    run = ET.SubElement(paragraph, W + "r")
    rpr = ET.SubElement(run, W + "rPr")
    ET.SubElement(rpr, W + "b")
    node = ET.SubElement(run, W + "t")
    node.text = "Algorithm 1. Adaptive Joint Training of JDST-DEC"
    return row


def format_table(table, table_index):
    tblpr = table.find(W + "tblPr")
    if tblpr is None:
        tblpr = ET.Element(W + "tblPr")
        table.insert(0, tblpr)

    alignment = get_or_add(tblpr, W + "jc")
    alignment.set(W + "val", "center")
    width = get_or_add(tblpr, W + "tblW")
    width.set(W + "type", "pct")
    width.set(W + "w", "5000")

    borders = get_or_add(tblpr, W + "tblBorders")
    set_table_border(borders, "top", "single", "12")
    set_table_border(borders, "bottom", "single", "12")
    set_table_border(borders, "insideH", "single", "4")
    set_table_border(borders, "left", "nil")
    set_table_border(borders, "right", "nil")
    set_table_border(borders, "insideV", "nil")

    rows = table.findall(W + "tr")
    if table_index == 0 and rows:
        table.insert(list(table).index(rows[0]), make_algorithm_title_row(rows[0]))

    for row_index, row in enumerate(table.findall(W + "tr")):
        for cell_index, cell in enumerate(row.findall(W + "tc")):
            tcpr = cell.find(W + "tcPr")
            if tcpr is None:
                tcpr = ET.Element(W + "tcPr")
                cell.insert(0, tcpr)
            valign = get_or_add(tcpr, W + "vAlign")
            valign.set(W + "val", "center")
            for paragraph in cell.iter(W + "p"):
                centered = table_index > 0 or row_index <= 1 or cell_index == 0
                set_paragraph_layout(
                    paragraph, first_line=False, centered=centered
                )
                ppr = paragraph.find(W + "pPr")
                if table_index == 0 and row_index > 1 and cell_index > 0:
                    jc = get_or_add(ppr, W + "jc")
                    jc.set(W + "val", "left")
                spacing = get_or_add(ppr, W + "spacing")
                spacing.set(W + "before", "40")
                spacing.set(W + "after", "40")


def reference_paragraph(template, segments):
    paragraph = ET.Element(W + "p")
    ppr = template.find(W + "pPr")
    if ppr is not None:
        paragraph.append(deepcopy(ppr))

    run_template = template.find(W + "r")
    rpr = run_template.find(W + "rPr") if run_template is not None else None

    def add_run(value, italic=False):
        run = ET.SubElement(paragraph, W + "r")
        if rpr is not None:
            run.append(deepcopy(rpr))
        if italic:
            props = run.find(W + "rPr")
            if props is None:
                props = ET.Element(W + "rPr")
                run.insert(0, props)
            ET.SubElement(props, W + "i")
            ET.SubElement(props, W + "iCs")
        node = ET.SubElement(run, W + "t")
        node.set(f"{{{XML_NS}}}space", "preserve")
        node.text = value

    for value, italic in segments:
        add_run(value, italic=italic)
    return paragraph


def main():
    with zipfile.ZipFile(BASE, "r") as base_zip:
        base_root = ET.fromstring(base_zip.read("word/document.xml"))
        base_rels_root = ET.fromstring(
            base_zip.read("word/_rels/document.xml.rels")
        )
        base_body = base_root.find(".//" + W + "body")
        base_children = list(base_body)

        method_start = next(
            i
            for i, element in enumerate(base_children)
            if element.tag == W + "p"
            and style(element) == "Heading1"
            and text(element) == "3. Proposed Method"
        )
        references_start = next(
            i
            for i, element in enumerate(base_children)
            if element.tag == W + "p"
            and style(element) == "Heading1"
            and text(element) == "References"
        )

        for element in base_children[method_start:references_start]:
            base_body.remove(element)

        added_media = {}
        with zipfile.ZipFile(METHOD, "r") as method_zip:
            method_root = ET.fromstring(method_zip.read("word/document.xml"))
            method_rels_root = ET.fromstring(
                method_zip.read("word/_rels/document.xml.rels")
            )
            method_body = method_root.find(".//" + W + "body")
            method_children = list(method_body)
            first_heading = next(
                i
                for i, element in enumerate(method_children)
                if element.tag == W + "p"
                and style(element) == "Heading1"
                and text(element) == "3. Proposed Method"
            )
            imported = [
                deepcopy(element)
                for element in method_children[first_heading:]
                if element.tag != W + "sectPr"
                and text(element) != "Algorithm 1. Adaptive joint training of JDST-DEC"
            ]

            referenced_ids = set()
            embed_attribute = f"{{{R_NS}}}embed"
            for element in imported:
                for node in element.iter():
                    if embed_attribute in node.attrib:
                        referenced_ids.add(node.get(embed_attribute))

            existing_ids = {
                rel.get("Id") for rel in base_rels_root
            }
            next_id = 1
            relation_map = {}
            method_relations = {
                rel.get("Id"): rel for rel in method_rels_root
            }
            for old_id in sorted(referenced_ids):
                relation = method_relations[old_id]
                while f"rId{next_id}" in existing_ids:
                    next_id += 1
                new_id = f"rId{next_id}"
                existing_ids.add(new_id)
                next_id += 1

                old_target = relation.get("Target")
                suffix = Path(old_target).suffix or ".png"
                new_target = f"media/method_{new_id}{suffix}"
                added_media[f"word/{new_target}"] = method_zip.read(
                    "word/" + old_target
                )

                new_relation = ET.SubElement(
                    base_rels_root, f"{{{PR_NS}}}Relationship"
                )
                new_relation.set("Id", new_id)
                new_relation.set("Type", relation.get("Type"))
                new_relation.set("Target", new_target)
                relation_map[old_id] = new_id

            for element in imported:
                for node in element.iter():
                    old_id = node.get(embed_attribute)
                    if old_id in relation_map:
                        node.set(embed_attribute, relation_map[old_id])

        table_index = 0
        for element in imported:
            strip_bookmarks(element)
            if element.tag == W + "p":
                format_body_paragraph(element)
            elif element.tag == W + "tbl":
                format_table(element, table_index)
                table_index += 1
        for offset, element in enumerate(imported):
            base_body.insert(method_start + offset, element)

        references = next(element for element in base_body if element.tag == W + "sdt")
        content = references.find(".//" + W + "sdtContent")
        ref_paragraphs = list(content.iter(W + "p"))
        additions = [
            (
                "[40]",
                [
                    ("[40] A. Kendall, Y. Gal, and R. Cipolla, “Multi-task learning using uncertainty to weigh losses for scene geometry and semantics,” in ", False),
                    ("Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition", True),
                    (", 2018, pp. 7482–7491.", False),
                ],
            ),
            (
                "[41]",
                [
                    ("[41] A. L. Maas et al., “Learning word vectors for sentiment analysis,” in ", False),
                    ("Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies", True),
                    (", 2011, pp. 142–150.", False),
                ],
            ),
            (
                "[42]",
                [
                    ("[42] X. Zhang, J. Zhao, and Y. LeCun, “Character-level convolutional networks for text classification,” in ", False),
                    ("Advances in Neural Information Processing Systems", True),
                    (", vol. 28, 2015, pp. 649–657.", False),
                ],
            ),
        ]
        current_references = text(references)
        for marker, segments in additions:
            if marker not in current_references:
                content.append(reference_paragraph(ref_paragraphs[-1], segments))

        updated_xml = ET.tostring(
            base_root, encoding="utf-8", xml_declaration=True
        )
        updated_rels = ET.tostring(
            base_rels_root, encoding="utf-8", xml_declaration=True
        )

        with tempfile.NamedTemporaryFile(
            dir=OUTPUT.parent, suffix=".docx", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with zipfile.ZipFile(temporary_path, "w") as output_zip:
                for item in base_zip.infolist():
                    if item.filename == "word/document.xml":
                        data = updated_xml
                    elif item.filename == "word/_rels/document.xml.rels":
                        data = updated_rels
                    else:
                        data = base_zip.read(item.filename)
                    output_zip.writestr(item, data)
                for filename, data in added_media.items():
                    output_zip.writestr(filename, data)
            temporary_path.chmod(0o644)
            temporary_path.replace(OUTPUT)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    print(OUTPUT)


if __name__ == "__main__":
    main()
