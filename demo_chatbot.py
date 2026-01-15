import streamlit as st
import pandas as pd
from io import BytesIO


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simple Excel Chatbot Demo",
    page_icon="💬",
    layout="wide"
)

# ── Title & Instructions ──────────────────────────────────────────────────────
st.title("💬 Simple Excel Chatbot Demo")
st.markdown("""
Upload one Excel file and start asking questions about its content!  
Examples:  
- "What are the column names?"  
- "Show me rows where [column] contains [value]"  
- "How many rows are there?"
""")

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your Excel file (.xlsx)",
    type=["xlsx"],
    help="Only one file for this demo"
)

# Session state to store the dataframe
if "df" not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    try:
        # Read the Excel file (first sheet by default)
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        
        st.success("File uploaded successfully!")
        
        # Show preview
        st.subheader("Data Preview (first 10 rows)")
        st.dataframe(df.head(10))
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Columns names", ", ".join(df.columns[:5]) + "..." if len(df.columns) > 5 else ", ".join(df.columns))
        
    except Exception as e:
        st.error(f"Error reading file: {e}")

# ── Chat interface ────────────────────────────────────────────────────────────
st.subheader("🧐 Ask questions about your data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something about the Excel file..."):
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.df is None:
            response = "Please upload an Excel file first! 📤"
        else:
            df = st.session_state.df
            prompt_lower = prompt.lower()
            
            # Very simple rule-based responses (you can expand this a lot)
            if "column" in prompt_lower and ("name" in prompt_lower or "columns" in prompt_lower):
                response = f"The columns in your file are:\n\n**{', '.join(df.columns)}**"
            
            elif "row" in prompt_lower and "many" in prompt_lower:
                response = f"Your file has **{len(df)} rows**."
            
            elif "show" in prompt_lower and "row" in prompt_lower:
                # Very basic filter example
                response = "Basic filtering is not fully implemented yet.\nTry more specific questions like 'What are the column names?' for now."
            
            else:
                # Fallback: search for any keyword in column names or first few rows
                found = False
                for col in df.columns:
                    if col.lower() in prompt_lower:
                        response = f"Column **{col}** exists!\n\nSample values:\n{df[col].head(5).to_string(index=False)}"
                        found = True
                        break
                
                if not found:
                    # Search in stringified data (very naive)
                    df_str = df.astype(str).to_string()
                    if any(word in df_str.lower() for word in prompt_lower.split()):
                        response = "I found something related in the data... but try to be more specific 😅"
                    else:
                        response = "Sorry, I couldn't find anything matching your question.\nTry asking about columns, row count, or specific column names!"
        
        st.markdown(response)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Optional: Clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
    
    
print("Demo chatbot app is running.")

print('Hello, World!')