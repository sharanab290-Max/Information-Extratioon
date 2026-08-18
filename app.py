import streamlit as st
import spacy
import pandas as pd
import networkx as nx

from pyvis.network import Network
from dateparser.search import search_dates
import streamlit.components.v1 as components
import re
import html




st.set_page_config(
    page_title="Information Extraction System",
    page_icon="🧠",
    layout="wide"
)




@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")


try:
    nlp = load_model()
except Exception:
    st.error(
        "spaCy English model is not installed. "
        "Run: python -m spacy download en_core_web_sm"
    )
    st.stop()



st.title(" Information Extraction System")

st.markdown(
    """
    ### NLP Pipeline

    This application performs:

    - Part-of-Speech (POS) Tagging
    - Named Entity Recognition (NER)
    - Relation Extraction
    - Event Extraction
    - Temporal Expression Detection
    - Temporal Event Ordering
    - Entity and Relation Tables
    - Interactive Graph Visualization
    """
)




st.sidebar.header(" Settings")

show_pos = st.sidebar.checkbox("POS Tagging", True)
show_entities = st.sidebar.checkbox("Named Entities", True)
show_relations = st.sidebar.checkbox("Relations", True)
show_events = st.sidebar.checkbox("Events", True)
show_temporal = st.sidebar.checkbox("Temporal Expressions", True)
show_graph = st.sidebar.checkbox("Relationship Graph", True)




sample_text = """
On January 15, 2025, Microsoft announced a new artificial intelligence
partnership with OpenAI in Seattle. Satya Nadella attended the event.
The company signed an agreement with OpenAI on January 20, 2025.
Microsoft later invested $10 billion in OpenAI.
"""

st.subheader(" Input Text")

text = st.text_area(
    "Enter or paste your text corpus:",
    value=sample_text,
    height=250
)




analyze = st.button(
    "🔍 Analyze Text",
    type="primary",
    use_container_width=True
)





def extract_pos(doc):
    """
    Extract Part-of-Speech information.
    """

    data = []

    for token in doc:

        if token.is_space:
            continue

        data.append({
            "Token": token.text,
            "Lemma": token.lemma_,
            "POS": token.pos_,
            "Detailed POS": token.tag_,
            "Dependency": token.dep_
        })

    return pd.DataFrame(data)


def extract_entities(doc):
    """
    Extract named entities.
    """

    data = []

    for ent in doc.ents:

        data.append({
            "Entity": ent.text,
            "Label": ent.label_,
            "Start": ent.start_char,
            "End": ent.end_char
        })

    return pd.DataFrame(data)


def extract_relations(doc):
    """
    Rule-based relation extraction.

    This identifies relationships between entities
    appearing in the same sentence.
    """

    relations = []

    for sent in doc.sents:

        entities = list(sent.ents)

        if len(entities) < 2:
            continue

        for i in range(len(entities) - 1):

            subject = entities[i]
            object_entity = entities[i + 1]

            relation = None

            sentence_lower = sent.text.lower()

            if "partnership" in sentence_lower:
                relation = "PARTNERSHIP_WITH"

            elif "agreement" in sentence_lower:
                relation = "AGREEMENT_WITH"

            elif "invested" in sentence_lower:
                relation = "INVESTED_IN"

            elif "acquired" in sentence_lower:
                relation = "ACQUIRED"

            elif "bought" in sentence_lower:
                relation = "BOUGHT"

            elif "founded" in sentence_lower:
                relation = "FOUNDED"

            elif "worked with" in sentence_lower:
                relation = "WORKED_WITH"

            elif "announced" in sentence_lower:
                relation = "ANNOUNCED"

            else:
                relation = "RELATED_TO"

            relations.append({
                "Subject": subject.text,
                "Relation": relation,
                "Object": object_entity.text,
                "Sentence": sent.text
            })

    return pd.DataFrame(relations)


def extract_events(doc):
    """
    Extract event-like structures using verbs.

    A more advanced version can later use
    Transformer-based event extraction.
    """

    events = []

    for sent in doc.sents:

        for token in sent:

            if token.pos_ != "VERB":
                continue

            subject = None
            object_entity = None

            for child in token.children:

                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child.subtree

                elif child.dep_ in ("dobj", "pobj", "obj"):
                    object_entity = child.subtree

            subject_text = " ".join(
                word.text for word in subject
            ) if subject else ""

            object_text = " ".join(
                word.text for word in object_entity
            ) if object_entity else ""

            events.append({
                "Event": token.lemma_,
                "Trigger": token.text,
                "Subject": subject_text,
                "Object": object_text,
                "Sentence": sent.text
            })

    return pd.DataFrame(events)


def extract_temporal_expressions(text):

    results = search_dates(
        text,
        settings={
            "RETURN_AS_TIMEZONE_AWARE": False
        }
    )

    data = []

    if results:

        for expression, parsed_date in results:

            data.append({
                "Expression": expression,
                "Normalized Date": parsed_date.strftime("%Y-%m-%d")
            })

    return pd.DataFrame(data)


def build_temporal_events(doc):

    timeline = []

    temporal_results = search_dates(
        doc.text,
        settings={
            "RETURN_AS_TIMEZONE_AWARE": False
        }
    )

    if not temporal_results:
        return pd.DataFrame()

    for expression, parsed_date in temporal_results:

        date_string = parsed_date.strftime("%Y-%m-%d")

        for sent in doc.sents:

            if expression.lower() in sent.text.lower():

                event_words = [
                    token.lemma_
                    for token in sent
                    if token.pos_ == "VERB"
                ]

                event = ", ".join(event_words)

                timeline.append({
                    "Date": date_string,
                    "Temporal Expression": expression,
                    "Event": event,
                    "Sentence": sent.text
                })

                break

    return pd.DataFrame(timeline)


def create_relation_graph(relations_df):

    graph = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#000000"
    )

    graph.barnes_hut()

    for _, row in relations_df.iterrows():

        subject = str(row["Subject"])
        relation = str(row["Relation"])
        object_entity = str(row["Object"])

        graph.add_node(
            subject,
            label=subject,
            title=subject
        )

        graph.add_node(
            object_entity,
            label=object_entity,
            title=object_entity
        )

        graph.add_edge(
            subject,
            object_entity,
            label=relation,
            title=relation
        )

    return graph


def annotate_text(doc):

    output = html.escape(doc.text)

    entities = sorted(
        doc.ents,
        key=lambda x: x.start_char,
        reverse=True
    )

    for ent in entities:

        label = html.escape(ent.label_)

        replacement = (
            f'<mark title="{label}">'
            f'{html.escape(ent.text)}'
            f' <small>({label})</small>'
            f'</mark>'
        )

        output = (
            output[:ent.start_char]
            + replacement
            + output[ent.end_char:]
        )

    return output




if analyze:

    if not text.strip():

        st.warning("Please enter some text.")

        st.stop()

    with st.spinner("Processing text..."):

        doc = nlp(text)

        pos_df = extract_pos(doc)
        entities_df = extract_entities(doc)
        relations_df = extract_relations(doc)
        events_df = extract_events(doc)
        temporal_df = extract_temporal_expressions(text)
        timeline_df = build_temporal_events(doc)

    st.success("Analysis completed successfully.")


    

    st.subheader( "Extraction Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Tokens",
        len([token for token in doc if not token.is_space])
    )

    col2.metric(
        "Entities",
        len(entities_df)
    )

    col3.metric(
        "Relations",
        len(relations_df)
    )

    col4.metric(
        "Events",
        len(events_df)
    )

    col5.metric(
        "Temporal",
        len(temporal_df)
    )


    # -----------------------------------------------------
    # ANNOTATED TEXT
    # -----------------------------------------------------

    st.subheader("📝 Annotated Text")

    annotated = annotate_text(doc)

    st.markdown(
        f"""
        <div style="
            padding:20px;
            border:1px solid #ddd;
            border-radius:10px;
            line-height:2;
            font-size:17px;
        ">
        {annotated}
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # POS
    # -----------------------------------------------------

    if show_pos:

        st.subheader("🔤 Part-of-Speech Tagging")

        st.dataframe(
            pos_df,
            use_container_width=True,
            hide_index=True
        )

        csv = pos_df.to_csv(index=False)

        st.download_button(
            "⬇️ Download POS Results",
            csv,
            "pos_results.csv",
            "text/csv"
        )


    # -----------------------------------------------------
    # NER
    # -----------------------------------------------------

    if show_entities:

        st.subheader("🏷️ Named Entities")

        if len(entities_df) > 0:

            st.dataframe(
                entities_df,
                use_container_width=True,
                hide_index=True
            )

            csv = entities_df.to_csv(index=False)

            st.download_button(
                " Download Entity Results",
                csv,
                "entities.csv",
                "text/csv"
            )

        else:

            st.info("No named entities detected.")


    

    if show_relations:

        st.subheader(" Extracted Relations")

        if len(relations_df) > 0:

            st.dataframe(
                relations_df,
                use_container_width=True,
                hide_index=True
            )

            csv = relations_df.to_csv(index=False)

            st.download_button(
                " Download Relation Results",
                csv,
                "relations.csv",
                "text/csv"
            )

        else:

            st.info("No relations detected.")


   

    if show_events:

        st.subheader(" Event Extraction")

        if len(events_df) > 0:

            st.dataframe(
                events_df,
                use_container_width=True,
                hide_index=True
            )

            csv = events_df.to_csv(index=False)

            st.download_button(
                "⬇️ Download Event Results",
                csv,
                "events.csv",
                "text/csv"
            )

        else:

            st.info("No events detected.")


    

    if show_temporal:

        st.subheader(" Temporal Expressions")

        if len(temporal_df) > 0:

            st.dataframe(
                temporal_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No temporal expressions detected.")


    

    st.subheader(" Event Timeline")

    if len(timeline_df) > 0:

        timeline_df = timeline_df.sort_values(
            by="Date"
        )

        st.dataframe(
            timeline_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Chronological Sequence")

        for _, row in timeline_df.iterrows():

            st.markdown(
                f"""
                **{row['Date']}**

                 **Event:** {row['Event']}

                 {row['Sentence']}
                """
            )

    else:

        st.info(
            "Timeline could not be constructed because "
            "no usable date-event combinations were detected."
        )


    

    if show_graph and len(relations_df) > 0:

        st.subheader("🕸️ Entity Relationship Graph")

        network = create_relation_graph(
            relations_df
        )

        network.save_graph(
            "relation_graph.html"
        )

        with open(
            "relation_graph.html",
            "r",
            encoding="utf-8"
        ) as file:

            graph_html = file.read()

        components.html(
            graph_html,
            height=650,
            scrolling=True
        )


    


