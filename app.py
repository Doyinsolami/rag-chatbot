import streamlit as st
from ingest import search
from generate import generate_answer
from patients import load_patient_directory, format_patient_label

st.set_page_config(page_title="Hospital Chatbot", page_icon="🏥")
st.title("Hospital Chatbot")
st.caption("Select a patient, then ask a question about their chart.")

directory = load_patient_directory()
labels = [format_patient_label(p) for p in directory]
label_to_patient = {format_patient_label(p): p for p in directory}

selected_label = st.selectbox(
    "Select patient",
    labels,
    index=None,
    placeholder="Search by name"
)

if selected_label:
    selected_patient = label_to_patient[selected_label]
    st.caption(
        f"Viewing chart for {selected_patient['name']}, patient ID {selected_patient['patient_id']}"
    )

    if st.session_state.get("active_patient") != selected_patient["patient_id"]:
        st.session_state.active_patient = selected_patient["patient_id"]
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask about this patient's chart")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Looking up the chart..."):
                results = search(
                    question,
                    n_results=5,
                    patient_id=selected_patient["patient_id"],
                )
                retrieved_docs = results["documents"][0]
                answer = generate_answer(question, retrieved_docs)

            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("Retrieved notes"):
            for i, doc in enumerate(retrieved_docs, start=1):
                st.markdown(f"**Result {i}**")
                st.write(doc)
else:
    st.info("Select a patient above to begin.")