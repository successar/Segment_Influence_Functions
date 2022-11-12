from typing import Optional
import streamlit as st
from matplotlib import cm as colormaps
from matplotlib import colors
from TextSelector import TextSelector

from ner_influence.entity_influence_indexing import Indexer

st.set_page_config(layout="wide")

#### Config
cmap = colormaps.get_cmap("Pastel2")
gcmap = colormaps.get_cmap("Set2")
get_color = lambda l: colors.rgb2hex(cmap(l - 1)) if l > 0 else "#FFFFFF"
get_gcolor = lambda l: colors.rgb2hex(gcmap(l - 1)) if l > 0 else "#FFFFFF"
####


# @st.cache(allow_output_mutation=True)
# def load_utils(ckpt_path):
#     scaffolding: NERScaffolding = ...
#     indexer = Indexer(scaffolding)
#     indexer.load_index()
#     indexer.generate_train_outputs()
#     indexer.generate_test_outputs("validation")
#     return indexer


# indexer = load_utils(output_dir)


def write_html(tokens, predicted_labels, gold_labels, bold_token_ids=None) -> str:
    html = []
    for i, (t, pl, gl) in enumerate(zip(tokens, predicted_labels, gold_labels)):
        pcolor = get_color(pl)
        gcolor = get_gcolor(gl)

        bold = "font-weight:bold" if bold_token_ids is not None and i in bold_token_ids else ""
        html.append(
            f'<span style="background-color:{pcolor};'
            f"border-bottom: 3px solid {gcolor};"
            f"border-top: 3px solid {gcolor};"
            f'border-right: 1px solid white;{bold}">{t} </span>'
        )

    return "".join(html)


def write_example(example, bold_token_ids=None) -> str:
    tokens = example["tokens"]
    predicted_labels = example["predicted_labels"]
    gold_labels = example["gold_labels"]

    # assert all(i in list(range(len(tokens))) for i in bold_token_ids), f"{len(tokens)} {bold_token_ids}"
    return write_html(tokens, predicted_labels, gold_labels, bold_token_ids)


def add_instance_selector(indexer, idxs):
    col1, col2 = st.columns(2)
    with col1:
        test_ids = sorted(list(indexer.test_outputs.keys())) if idxs is None else idxs
        idx = st.number_input(label="Select test example", min_value=0, max_value=len(test_ids) - 1)
        idx = test_ids[idx]

    class_names = indexer.scaffolding.class_names
    with col2:
        st.write(
            "Color Code: "
            + write_html(class_names, list(range(len(class_names))), list(range(len(class_names))))
            + "<br/> Predicted: Fill, Gold: Border",
            unsafe_allow_html=True,
        )
        st.write(f"Total ids : {len(test_ids)}")

    st.write(
        write_example(
            indexer.test_outputs[idx],
        ),
        unsafe_allow_html=True,
    )

    return idx


def add_selector(indexer: Indexer, idxs: Optional[list[str]] = None):
    col1, col2 = st.columns(2)
    with col1:
        test_ids = sorted(list(indexer.test_outputs.keys())) if idxs is None else idxs
        idx = st.number_input(label="Select test example", min_value=0, max_value=len(test_ids) - 1)
        idx = test_ids[idx]

    class_names = indexer.scaffolding.class_names
    with col2:
        st.write(
            "Color Code: "
            + write_html(class_names, list(range(len(class_names))), list(range(len(class_names))))
            + "<br/> Predicted: Fill, Gold: Border",
            unsafe_allow_html=True,
        )
        st.write(f"Total ids : {len(test_ids)}")

    word_index = TextSelector(
        tokens=indexer.test_outputs[idx]["tokens"],
        colors=[get_color(label) for label in indexer.test_outputs[idx]["predicted_labels"]],
        border_colors=[get_gcolor(label) for label in indexer.test_outputs[idx]["gold_labels"]],
    )
    word_index = word_index["selected"] if word_index is not None else 0
    return idx, word_index


def display_supporters_opposers(indexer: Indexer, idx: str, word_index: int):
    supporters, opposers = indexer.search(idx, word_index, k=5)
    support_col, oppose_col = st.columns(2)

    with support_col:
        st.subheader(f"Support the Label:")
        for sentence_id, token_id, distance in supporters:
            st.write(
                write_example(indexer.train_outputs[sentence_id], bold_token_ids=[token_id]),
                unsafe_allow_html=True,
            )
            st.write("<hr>", unsafe_allow_html=True)

    with oppose_col:
        st.subheader(f"Oppose the Label:")
        for sentence_id, token_id, distance in opposers:
            st.write(
                write_example(indexer.train_outputs[sentence_id], bold_token_ids=[token_id]),
                unsafe_allow_html=True,
            )
            st.write("<hr>", unsafe_allow_html=True)


def display_instance_supporters_opposers(indexer: Indexer, idx: str):
    supporters, opposers = indexer.search(idx, k=5)
    support_col, oppose_col = st.columns(2)

    with support_col:
        st.subheader(f"Support the Gold Label:")
        for sentence_id, distance in supporters:
            st.write(write_example(indexer.train_outputs[sentence_id]), unsafe_allow_html=True)
            st.write("<hr>", unsafe_allow_html=True)

    with oppose_col:
        st.subheader(f"Oppose the Gold Label:")
        for sentence_id, distance in opposers:
            st.write(write_example(indexer.train_outputs[sentence_id]), unsafe_allow_html=True)
            st.write("<hr>", unsafe_allow_html=True)
