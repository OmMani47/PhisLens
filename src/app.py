"""Local Streamlit demo for PhishLens.

Run from the repository root:

    streamlit run src/app.py

The page never sends the message anywhere. It loads the saved scikit-learn
pipeline and shows the verdict, the security checks, and the wording the
linear model used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from src.paths import METRICS_PATH
from src.pipeline import load_model
from src.predict import classify_email, load_demos

st.set_page_config(
    page_title="PhishLens",
    page_icon="✉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px;}
    .phishlens-kicker {letter-spacing: 0.14em; text-transform: uppercase;
        font-size: 0.78rem; color: #8A6A2F; margin-bottom: 0.2rem;}
    div[data-testid="stMetric"] {background: #EBE4D4; padding: 0.6rem 0.8rem; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_model():
    return load_model()


@st.cache_data
def cached_demos() -> list[dict]:
    return load_demos()


@st.cache_data
def cached_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _matching_demo(demos: list[dict]) -> dict | None:
    for demo in demos:
        if (
            st.session_state.get("sender", "") == demo.get("sender", "")
            and st.session_state.get("reply_to", "") == demo.get("reply_to", "")
            and st.session_state.get("subject", "") == demo.get("subject", "")
            and st.session_state.get("body", "") == demo.get("body", "")
        ):
            return demo
    return None


def _clear_fields() -> None:
    st.session_state.sender = ""
    st.session_state.reply_to = ""
    st.session_state.subject = ""
    st.session_state.body = ""
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.do_classify = False


def _load_demo(demo_id: str) -> None:
    demo = next(item for item in cached_demos() if item["id"] == demo_id)
    st.session_state.sender = demo.get("sender", "")
    st.session_state.reply_to = demo.get("reply_to", "")
    st.session_state.subject = demo.get("subject", "")
    st.session_state.body = demo.get("body", "")
    st.session_state.do_classify = True


def _cue_frame(cues: list[dict], toward: str) -> pd.DataFrame | None:
    if not cues:
        return None
    return pd.DataFrame(
        [
            {
                "Cue": cue["feature"],
                "Pulls toward": toward,
                "Contribution": float(cue["contribution"]),
            }
            for cue in cues
        ]
    )


def _render_result(result: dict, demo: dict | None) -> None:
    phishing = result["label"] == "phishing"
    probability = result["phishing_probability"]
    if phishing:
        st.error(f"Verdict: phishing. Model probability {probability:.0%}.")
    else:
        st.success(
            f"Verdict: legitimate. Model probability {result['legitimate_probability']:.0%} "
            f"({probability:.0%} phishing)."
        )
    if result["low_margin"]:
        st.warning("The margin is thin. Read the message before you trust this verdict.")

    if demo is not None:
        classroom = demo["classroom_label"]
        agrees = classroom == result["label"]
        if agrees:
            st.caption(f"Classroom label: {classroom}. The model agrees. {demo['lesson']}")
        else:
            st.warning(
                f"Classroom label: {classroom}. The model said {result['label']}. {demo['lesson']}"
            )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Phishing probability", f"{probability:.0%}")
    metric_cols[1].metric("Legitimate probability", f"{result['legitimate_probability']:.0%}")
    metric_cols[2].metric("Security checks fired", str(len(result["flags"])))
    st.write(result["summary"])

    signal_col, word_col = st.columns(2, gap="large")
    with signal_col:
        st.subheader("Security checks")
        if result["flags"]:
            for flag in result["flags"]:
                st.markdown(f"**{flag['title']}**")
                st.write(flag["detail"])
        else:
            st.info("None of the handcrafted checks fired. The verdict is coming from the wording weights.")
    with word_col:
        st.subheader("Wording the model used")
        st.caption("Contribution is the TF-IDF value times the logistic-regression weight. Positive pushes toward phishing.")
        phishing_cues = _cue_frame(result["top_phishing_cues"], "phishing")
        legit_cues = _cue_frame(result["top_legitimate_cues"], "legitimate")
        if phishing_cues is None and legit_cues is None:
            st.info("No word cue cleared the vocabulary. Character n-grams may still have contributed.")
        if phishing_cues is not None:
            st.markdown("**Toward phishing**")
            st.dataframe(phishing_cues, hide_index=True, use_container_width=True)
        if legit_cues is not None:
            st.markdown("**Toward legitimate**")
            st.dataframe(legit_cues, hide_index=True, use_container_width=True)

    with st.expander("Security-feature weights on this email"):
        st.caption(
            "These are the handcrafted checks that were non-zero, with the weight learned on the synthetic set. "
            "A check can fire and still carry a small weight."
        )
        rows = result["heuristic_contributions"]
        if rows:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Feature": row["feature"],
                            "Value": row["value"],
                            "Weight": row["coefficient"],
                            "Contribution": row["contribution"],
                        }
                        for row in rows
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("No handcrafted feature was above zero.")

    with st.expander("Character n-grams"):
        st.caption("Character pieces catch odd spellings such as a brand with a digit swapped in. They are noisier than words.")
        char_phish = _cue_frame(result["top_phishing_char_cues"], "phishing")
        char_legit = _cue_frame(result["top_legitimate_char_cues"], "legitimate")
        if char_phish is not None:
            st.dataframe(char_phish, hide_index=True, use_container_width=True)
        if char_legit is not None:
            st.dataframe(char_legit, hide_index=True, use_container_width=True)
        if char_phish is None and char_legit is None:
            st.write("No character cue stood out for this message.")


def _sidebar(metrics: dict | None) -> None:
    with st.sidebar:
        st.header("How a score is made")
        st.write(
            "PhishLens stacks two views of the same email. One is a short list of security checks: "
            "links, sender identity, urgency, and requests for a password or a payment. The other is "
            "TF-IDF over words and character spans. Logistic regression adds them up."
        )
        st.write(
            "A positive weight leans phishing when that feature is present. The page shows both the "
            "checks that fired and the wording that moved the score."
        )
        st.header("Synthetic test scores")
        if metrics is None:
            st.info("No metrics file yet. Run python -m src.train.")
        else:
            st.caption(metrics["disclaimer"])
            score_cols = st.columns(2)
            score_cols[0].metric("Phishing F1", f"{metrics['f1_phishing']:.3f}")
            score_cols[1].metric("Accuracy", f"{metrics['accuracy']:.3f}")
            matrix = metrics["confusion_matrix"]["matrix"]
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["True legitimate", "True phishing"],
                    columns=["Pred. legitimate", "Pred. phishing"],
                ),
                use_container_width=True,
            )
            st.caption(
                f"Stratified holdout, {metrics['test_rows']} of {metrics['rows']} synthetic emails, "
                f"seed {metrics['seed']}."
            )
        st.header("Use")
        st.write(
            "This is a classroom demo for defensive practice. It is not a mail filter, and it is not "
            "permission to write or send phishing. Paste only mail you are allowed to examine."
        )


def main() -> None:
    try:
        model = cached_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    demos = cached_demos()
    metrics = cached_metrics()
    _sidebar(metrics)

    for key, value in {
        "sender": "",
        "reply_to": "",
        "subject": "",
        "body": "",
        "do_classify": False,
        "result": None,
        "error": None,
    }.items():
        st.session_state.setdefault(key, value)

    st.markdown('<p class="phishlens-kicker">Defensive teaching demo</p>', unsafe_allow_html=True)
    st.title("PhishLens")
    st.write(
        "A local classifier for a cybersecurity and AI portfolio. Paste a message, or load a "
        "synthetic classroom example, and read why the model leaned phishing or legitimate."
    )
    st.caption(
        "Examples are fiction. Links use the reserved .example suffix or documentation "
        "IP addresses. Do not treat a verdict as a reason to click, reply, or pay."
    )

    pending = st.session_state.do_classify
    st.session_state.do_classify = False

    st.subheader("Load a classroom example")
    groups = [
        ("phish", "Signals a detector should notice"),
        ("legit", "Ordinary mail"),
        ("edge", "A case the model gets wrong on purpose"),
    ]
    for group, label in groups:
        members = [demo for demo in demos if demo["group"] == group]
        if not members:
            continue
        st.caption(label)
        columns = st.columns(2)
        for index, demo in enumerate(members):
            columns[index % 2].button(
                demo["title"],
                key=f"demo-{demo['id']}",
                use_container_width=True,
                on_click=_load_demo,
                args=(demo["id"],),
            )

    st.subheader("Message")
    st.text_input("From", key="sender", placeholder="Name <user@host>")
    st.text_input("Reply-To", key="reply_to", placeholder="Optional. Leave blank if the mail has none.")
    st.text_input("Subject", key="subject")
    st.text_area("Body", key="body", height=220)

    action_col, clear_col = st.columns(2)
    clicked = action_col.button("Classify email", type="primary", use_container_width=True)
    clear_col.button("Clear", use_container_width=True, on_click=_clear_fields)
    if clicked or pending:
        try:
            st.session_state.result = classify_email(
                {
                    "sender": st.session_state.sender,
                    "reply_to": st.session_state.reply_to,
                    "subject": st.session_state.subject,
                    "body": st.session_state.body,
                },
                pipeline=model,
            )
            st.session_state.error = None
        except ValueError as exc:
            st.session_state.result = None
            st.session_state.error = str(exc)

    st.subheader("Verdict")
    if st.session_state.error:
        st.warning(st.session_state.error)
    elif st.session_state.result is None:
        st.info(
            "Classify a message to see the verdict, the probability, the security checks that fired, "
            "and the words that moved the linear model. Nothing you type leaves this process."
        )
    else:
        _render_result(st.session_state.result, _matching_demo(demos))


# Streamlit reruns this file from top to bottom on every interaction.
main()
