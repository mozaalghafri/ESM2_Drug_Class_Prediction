import streamlit as st
import torch
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = str(Path(__file__).resolve().parents[1] / "final_drug_target_model")
TOKENIZER_NAME = "facebook/esm2_t6_8M_UR50D"

label_map = {
    0: "kinase",
    1: "GPCR",
    2: "ion channel"
}

class_info = {
    "kinase": {
        "description": "Kinases are enzymes that transfer phosphate groups to proteins. They are important in cell signalling, growth, metabolism, and cancer biology.",
        "drug_relevance": "Many targeted cancer therapies are kinase inhibitors."
    },
    "GPCR": {
        "description": "GPCRs are membrane receptors that detect external signals and activate intracellular signalling pathways.",
        "drug_relevance": "GPCRs are one of the most important drug target families, especially in neurological, cardiovascular, and metabolic diseases."
    },
    "ion channel": {
        "description": "Ion channels are membrane proteins that allow ions such as Na⁺, K⁺, Ca²⁺, or Cl⁻ to move across cell membranes.",
        "drug_relevance": "They are important drug targets in pain, epilepsy, heart rhythm disorders, and neurological conditions."
    }
}

example_sequences = {
    "None": "",
    "Kinase example": "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP",
    "GPCR example": "MNGTEGPNFYVPFSNATGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTRAEGFNNESFVIYMFVVHFIIPLIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWLPYAGVAFYIFTHQGSNFGPIFMTIPAFFAKSAAIYNPVIYIMMNKQFRNCMLTTICCGKNPLGDDEASTTVSKTETSQVAPA",
    "Ion channel example": "MFPTGWRPKLSESIAASRMLWQPMAAVAVVQIGLLWFSPPVWGQDMVSPPPPIADEPLTVNTGIYLIECYYSLDDKAETFKVNAFLSLSWKDRRLAFDPVRSGVRVKTYEPEAIWIPEIRVFNVENARDADVDISVSPDGTYQYLERFSARVLSPLDFRRYPFDSQTLHIYLIVRSVDTRNIVLAVDLEKVGKNDDVFLTGWDIESFTAVVKPANFALEDRLESKLDYQLRISRQYFSYIPNIILPMLFILFISWTAFWSTSYEANVTLVVSTLIAHIAFNILVETNLPKTPYMTYTGAIIFMIYLFYFVA"
}

valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(text):
    lines = text.strip().splitlines()
    sequence_lines = [line.strip() for line in lines if not line.startswith(">")]
    return "".join(sequence_lines).replace(" ", "").upper()


def clean_sequence(seq):
    return seq.replace(" ", "").replace("\n", "").replace("\r", "").upper()


@st.cache_data
def fetch_uniprot_entry(accession):
    accession = accession.strip().upper()
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"

    response = requests.get(url, timeout=20)

    if response.status_code != 200:
        return None

    data = response.json()

    protein_name = (
        data.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "Not available")
    )

    organism = data.get("organism", {}).get("scientificName", "Not available")
    sequence = data.get("sequence", {}).get("value", "")

    comments = data.get("comments", [])
    function_text = "Not available"

    for comment in comments:
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                function_text = texts[0].get("value", "Not available")
                break

    return {
        "accession": accession,
        "protein_name": protein_name,
        "organism": organism,
        "function": function_text,
        "sequence": sequence
    }


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model.eval()
    return tokenizer, model


def predict_sequence(sequence, tokenizer, model):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    prediction = int(np.argmax(probabilities))
    predicted_class = label_map[prediction]

    return predicted_class, probabilities


st.title("Drug Target Class Predictor")

st.write(
    "Predict whether a protein sequence is most likely a kinase, GPCR, or ion channel."
)

tokenizer, model = load_model()

if "uniprot_sequence" not in st.session_state:
    st.session_state.uniprot_sequence = ""

if "uniprot_info" not in st.session_state:
    st.session_state.uniprot_info = None

st.subheader("Input method")

input_method = st.radio(
    "Choose how you want to provide the protein sequence:",
    ["Paste sequence", "Upload FASTA file", "UniProt accession lookup"]
)

sequence = ""

if input_method == "Paste sequence":
    selected_example = st.selectbox(
        "Optional: choose an example sequence",
        list(example_sequences.keys())
    )

    default_sequence = example_sequences[selected_example]

    sequence = st.text_area(
        "Protein Sequence",
        value=default_sequence,
        height=200
    )

elif input_method == "Upload FASTA file":
    uploaded_file = st.file_uploader(
        "Upload a FASTA file",
        type=["fasta", "fa", "txt"]
    )

    if uploaded_file is not None:
        fasta_text = uploaded_file.read().decode("utf-8")
        sequence = parse_fasta(fasta_text)

        sequence = st.text_area(
            "Extracted sequence",
            value=sequence,
            height=200
        )

elif input_method == "UniProt accession lookup":
    accession = st.text_input(
        "Enter UniProt accession ID",
        placeholder="Example: P00533"
    )

    if st.button("Fetch from UniProt"):
        if accession.strip() == "":
            st.warning("Please enter a UniProt accession ID.")

        else:
            uniprot_info = fetch_uniprot_entry(accession)

            if uniprot_info is None:
                st.error("Could not find this UniProt entry.")

            else:
                st.session_state.uniprot_info = uniprot_info
                st.session_state.uniprot_sequence = uniprot_info["sequence"]
                st.success("UniProt entry found.")

    if st.session_state.uniprot_info is not None:
        info = st.session_state.uniprot_info

        st.write(f"**Protein name:** {info['protein_name']}")
        st.write(f"**Organism:** {info['organism']}")
        st.write(f"**Function:** {info['function']}")

    sequence = st.text_area(
        "Sequence from UniProt",
        value=st.session_state.uniprot_sequence,
        height=200
    )

cleaned_sequence = clean_sequence(sequence)

invalid_chars = sorted(set(cleaned_sequence) - valid_amino_acids)

if cleaned_sequence and invalid_chars:
    st.warning(
        f"Invalid amino acid letters found: {', '.join(invalid_chars)}. Please use standard one-letter amino acid codes only."
    )

if st.button("Predict class"):
    if cleaned_sequence.strip() == "":
        st.warning("Please provide a protein sequence first.")

    elif invalid_chars:
        st.error("Prediction stopped because the sequence contains invalid characters.")

    else:
        predicted_class, probabilities = predict_sequence(
            cleaned_sequence,
            tokenizer,
            model
        )

        st.success(f"Predicted class: {predicted_class}")

        st.subheader("Class information")
        st.write(class_info[predicted_class]["description"])

        st.subheader("Why this matters as a drug target")
        st.write(class_info[predicted_class]["drug_relevance"])

        st.subheader("Prediction confidence")

        confidence_df = pd.DataFrame({
            "Class": [label_map[i] for i in range(len(probabilities))],
            "Confidence": probabilities
        })

        st.bar_chart(
            confidence_df,
            x="Class",
            y="Confidence"
        )

        for i, prob in enumerate(probabilities):
            st.write(f"{label_map[i]}: {prob:.2%}")