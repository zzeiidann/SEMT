from copy import deepcopy
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile


SOURCE = Path("/Users/mraffyzeidan/Downloads/SEMT.docx")
OUTPUT = Path("/Users/mraffyzeidan/Downloads/SEMT/SEMT_updated_introduction.docx")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
W = f"{{{W_NS}}}"
ET.register_namespace("w", W_NS)


INTRODUCTION = [
    (
        "The rapid growth of digital platforms—including social media, review websites, "
        "and news portals—has produced an enormous volume of unstructured textual data. "
        "Every day, users generate text containing opinions, emotions, and perspectives on "
        "a wide range of issues. These data offer valuable opportunities for analyzing public "
        "opinion and social trends, but their scale makes manual analysis impractical. "
        "Computational approaches based on *text mining* and *Natural Language Processing* "
        "(NLP) are therefore required to process and utilize this information effectively."
    ),
    (
        "Sentiment analysis is a major and continually developing area of NLP. It aims to "
        "identify and classify the opinions and emotions expressed in text [35]. In a "
        "*supervised learning* setting, sentiment analysis is formulated as a classification "
        "problem and therefore requires sentiment-labeled training data. One widely used "
        "formulation is *document-level sentiment analysis*, which assumes that each document "
        "has one dominant overall polarity, such as positive, negative, or neutral. This "
        "formulation is comparatively simple to model and efficient to annotate because each "
        "document requires only one sentiment label."
    ),
    (
        "Nevertheless, document-level sentiment analysis can overlook detailed information "
        "because it assumes a uniform emotional orientation throughout a document. A document "
        "may discuss several aspects or topics with different sentiment polarities. This "
        "limitation motivated topic-level sentiment approaches, which identify sentiment "
        "toward particular topics. Although these approaches can provide finer-grained "
        "analysis, they also require more complex annotation and modeling."
    ),
    (
        "Early joint sentiment-topic models attempted to learn thematic and affective "
        "information simultaneously. The *Joint Sentiment–Topic* (JST) model extended "
        "*Latent Dirichlet Allocation* (LDA) by incorporating a sentiment dimension into a "
        "single probabilistic model [27], [1]. However, these probabilistic approaches rely "
        "on bag-of-words representations and consequently have limited ability to capture "
        "deep semantic relationships among words. More recent neural topic models and "
        "transformer-based methods provide richer contextual representations; BERTopic, for "
        "example, combines transformer embeddings with clustering to discover semantically "
        "coherent topics [13]."
    ),
    (
        "Because fine-grained topic-level sentiment analysis is costly, document-level "
        "sentiment analysis can instead be enhanced by incorporating topic information into "
        "representation learning. Two general strategies can be distinguished. In a separate "
        "document-level sentiment-topic approach, topic detection and sentiment classification "
        "are performed by independent models. In a joint document-level sentiment-topic "
        "approach, both tasks are optimized within a unified framework so that topic and "
        "sentiment information can interact during training."
    ),
    (
        "A deep-learning implementation of joint sentiment classification and topic detection "
        "was introduced by Gui et al. [36]. Their multi-task mutual-learning model uses a "
        "shared representation derived from a hierarchical attention network for sentiment "
        "classification, while topic modeling processes that representation through a "
        "*variational autoencoder* before applying K-Means. The approach improves both tasks, "
        "but topic clusters are still formed after the principal representation has been "
        "learned. Consequently, the relationship between the latent representation and the "
        "cluster structure is not optimized directly throughout training."
    ),
    (
        "Autoencoders are commonly used to reduce dimensionality while retaining important "
        "information from the input. *Deep Embedded Clustering* (DEC) integrates clustering "
        "directly into an autoencoder's latent space, allowing representations and cluster "
        "assignments to be refined jointly [25]. This produces a compact latent space whose "
        "structure is explicitly optimized for clustering and offers a suitable foundation "
        "for coupling topic discovery with sentiment classification."
    ),
    (
        "Building on these developments, this study proposes a *Joint Document-Level "
        "Sentiment–Topic* model based on *Deep Embedded Clustering* (JDST-DEC). Frozen BERT "
        "representations [7] are projected into a latent space through a shared encoder. A "
        "DEC-based clustering head and a sentiment-classification head are then trained "
        "simultaneously. Unlike approaches that separate clustering from classification, "
        "JDST-DEC enables the latent space to support sentiment prediction while developing "
        "a more structured and interpretable topic organization. Topic discovery is treated "
        "as an auxiliary task that regularizes the shared representation rather than as the "
        "primary output."
    ),
    (
        "The proposed model is evaluated on the IMDB and Yelp review datasets using several "
        "numbers of topics (K = 30, 50, and 80). Sentiment performance is assessed through "
        "accuracy, precision, recall, and F1-score, whereas topic quality is evaluated through "
        "NPMI-based topic coherence and topic diversity. JDST-DEC is compared with NN-BERT, "
        "the multi-task mutual-learning model of Gui et al. [36], BERTopic-HDBSCAN, and "
        "BERTopic-KMeans."
    ),
    (
        "The remainder of this paper is organized as follows. Section 2 reviews the relevant "
        "literature; Section 3 describes the proposed methodology; Section 4 presents the "
        "experimental results and discussion; and Section 5 concludes the paper and outlines "
        "directions for future research."
    ),
]


def paragraph_text(element):
    return "".join(node.text or "" for node in element.iter(W + "t")).strip()


def make_paragraph(template, text):
    paragraph = ET.Element(W + "p")
    properties = template.find(W + "pPr")
    if properties is not None:
        paragraph.append(deepcopy(properties))

    run_template = template.find(W + "r")
    run_properties = run_template.find(W + "rPr") if run_template is not None else None
    parts = re.split(r"(\*[^*]+\*)", text)
    for part in parts:
        if not part:
            continue
        italic = part.startswith("*") and part.endswith("*")
        value = part[1:-1] if italic else part
        run = ET.SubElement(paragraph, W + "r")
        if run_properties is not None:
            run.append(deepcopy(run_properties))
        if italic:
            rpr = run.find(W + "rPr")
            if rpr is None:
                rpr = ET.Element(W + "rPr")
                run.insert(0, rpr)
            ET.SubElement(rpr, W + "i")
            ET.SubElement(rpr, W + "iCs")
        node = ET.SubElement(run, W + "t")
        node.set(f"{{{XML_NS}}}space", "preserve")
        node.text = value
    return paragraph


def make_blank(template):
    paragraph = ET.Element(W + "p")
    properties = template.find(W + "pPr")
    if properties is not None:
        paragraph.append(deepcopy(properties))
    return paragraph


def update_document_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    body = root.find(".//" + W + "body")
    children = list(body)

    intro_index = next(
        i for i, element in enumerate(children)
        if element.tag == W + "p" and paragraph_text(element) == "Introduction"
    )
    related_index = next(
        i for i, element in enumerate(children)
        if element.tag == W + "p" and paragraph_text(element) == "Related Works"
    )
    body_template = next(
        element for element in children[intro_index + 1:related_index]
        if element.tag == W + "p" and paragraph_text(element)
    )

    for element in children[intro_index + 1:related_index]:
        body.remove(element)

    position = intro_index + 1
    body.insert(position, make_blank(body_template))
    position += 1
    for paragraph in INTRODUCTION:
        body.insert(position, make_paragraph(body_template, paragraph))
        position += 1
        body.insert(position, make_blank(body_template))
        position += 1

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE, "r") as source:
        updated_xml = update_document_xml(source.read("word/document.xml"))
        with tempfile.NamedTemporaryFile(
            dir=OUTPUT.parent, suffix=".docx", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with zipfile.ZipFile(temporary_path, "w") as target:
                for item in source.infolist():
                    data = (
                        updated_xml
                        if item.filename == "word/document.xml"
                        else source.read(item.filename)
                    )
                    target.writestr(item, data)
            temporary_path.replace(OUTPUT)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    print(OUTPUT)


if __name__ == "__main__":
    main()
