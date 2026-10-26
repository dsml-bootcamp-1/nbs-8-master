![Ironhack logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Streamlit: From Notebook to App

*Data Science & Machine Learning - Week 8, Day 3*

## How to use this lesson

A notebook is for you: it mixes code, outputs and notes, and only someone who can run Python can use it. A colleague in customer care, a manager or a client needs something else: a page in the browser with a few inputs, a button and a clear result. **Streamlit** turns a Python script into exactly that, with no HTML, CSS or JavaScript.

Today you build six small apps, each one adding one idea:

| Step | App | What you learn |
|---|---|---|
| 1 | `apps/01_hello.py` | how a Streamlit script runs, and reruns |
| 2 | `apps/02_widgets_and_layout.py` | widgets, columns, sidebar, tabs, forms and session state |
| 3 | `apps/03_data_and_charts.py` | a dashboard: filters, metrics, charts, and caching data |
| 4 | `apps/04_model_app.py` | a saved scikit-learn model behind a form, and caching resources |
| 5 | `apps/05_chatbot.py` | a chatbot with memory and streaming, and keeping keys secret |
| 6 | `apps/06_rag_app.py` | Friday's RAG pipeline as an app, with its sources |

Then you see how to split an app into pages, **deploy** it for free on Streamlit Community Cloud, **test** it without a browser, and publish it responsibly. Read each section, open the matching file, run it, and change something. The code is short on purpose: read all of it.

## Setup

**Download or clone the whole `s-806` folder**, and run every command **from inside it** (the apps read `data/` and `models/` with paths relative to it).

```bash
pip install streamlit                 # the only new library; the others are from Weeks 5-8
python train_churn_model.py           # once, for step 4: saves models/churn_pipeline.joblib
streamlit run apps/01_hello.py        # opens http://localhost:8501 in your browser
```

Stop an app with `Ctrl+C` in the terminal. Steps 5 and 6 need your OpenAI key, in the same `.env` file as last week (`OPENAI_API_KEY=sk-...`), next to this README. Steps 5 and 6 cost a fraction of a cent per question.

[visual: a terminal running `streamlit run apps/01_hello.py` on the left, the browser showing the app on the right]

---

## 1. How a Streamlit app runs

Open `apps/01_hello.py`. It is an ordinary Python script: `st.title`, `st.write`, `st.text_input`, `st.button`, `st.metric` each draw something on the page, in the order they appear.

The one idea to understand before anything else: **every time the user touches a widget, Streamlit runs the whole script again, from the first line to the last.** Widgets return their current value: `st.text_input` returns the text in the box, and `st.button` returns `True` **only in the run triggered by the click**, and `False` in every other run.

```python
name = st.text_input("Your name")
if st.button("Greet"):
    st.success(f"Hello {name}!")
```

So when the user types *Ana*, the script reruns, `name` is `"Ana"`, the button is `False`, and nothing is greeted. When they click, the script reruns once more with the button `True`, and the page shows *Hello Ana!*. Change something in the file and save: the app offers to rerun with your new code.

This rerun model is why Streamlit is so simple (no callbacks, no events: just a script) and why the next steps need **caching** and **session state**: without them, everything would be recomputed and forgotten at every click.

## 2. Widgets, layout and session state

`apps/02_widgets_and_layout.py` shows the building blocks you will use most:

- **Inputs:** `st.selectbox`, `st.slider`, `st.checkbox`, `st.radio`, `st.multiselect`, `st.number_input`, `st.text_area`, `st.file_uploader`.
- **Layout:** `st.sidebar` for settings, `st.columns` to put things side by side, `st.tabs` for separate views, `st.expander` for details on demand.
- **Forms:** inside `with st.form(...)`, widgets do not trigger a rerun until the user presses `st.form_submit_button`. Use a form when the user fills several fields before acting, as in the model app of step 4.
- **Session state:** a normal variable is reset at every rerun. `st.session_state` is a dictionary that survives reruns for as long as the browser tab is open, so it can count clicks, hold a conversation, or remember a result.

```python
if "clicks" not in st.session_state:      # initialise once
    st.session_state.clicks = 0
if st.button("Click me"):
    st.session_state.clicks += 1          # survives the next rerun
```

## 3. A data dashboard, and caching data

`apps/03_data_and_charts.py` is a dashboard over the churn data from Weeks 4-6: filters in the sidebar, three `st.metric` numbers, a built-in `st.bar_chart`, a Matplotlib figure shown with `st.pyplot`, the filtered table with `st.dataframe`, and a `st.download_button`.

Reading the CSV at every rerun would be wasteful, so the loading function is decorated with **`@st.cache_data`**: the first run reads the file, and later runs reuse the result, as long as the arguments are the same.

```python
@st.cache_data
def load_data(path="data/churn.csv"):
    return pd.read_csv(path)
```

Try the filters: with at least 4 calls to customer service, the churn rate jumps from about 15% to about 50%, the same finding as in Week 4, but now anyone in the company can explore it. Notice `st.stop()` when no customer matches the filters: it ends the run cleanly instead of showing empty charts or an error.

## 4. A model behind a form, and caching resources

This is the step that turns your Project 2 or Project 3 model into a product. First run `python train_churn_model.py`: it trains a random forest pipeline on the churn data (test ROC AUC about 0.89) and saves it with `joblib`, as on Tuesday, together with the typical value of every column.

`apps/04_model_app.py` loads the model once with **`@st.cache_resource`**, collects six inputs in a form, fills the other columns with typical values, and shows the probability of churn with `st.metric` and `st.progress`.

| Decorator | Use it for | What it returns |
|---|---|---|
| `@st.cache_data` | data: DataFrames, query results, API responses | a fresh **copy** for every caller |
| `@st.cache_resource` | resources: models, database connections, API clients, vector indexes | the **same object**, shared by every rerun and every user |

Without `cache_resource`, the model would be loaded from disk at every click; with a 300 MB transformer, the app would be unusable.

Two details matter in every model app. The input must have **exactly the columns the pipeline was trained on**, which is why the app starts from the saved typical values and overwrites the ones the user chose. And the output must make sense to the user: a probability, a clear decision (*offer a retention call*), and the details in an expander for whoever wants them.

## 5. A chatbot with memory and streaming

`apps/05_chatbot.py` is a support chatbot in 40 lines:

- `st.chat_input` at the bottom of the page and `st.chat_message` bubbles for each turn.
- The conversation lives in `st.session_state.messages`, and is redrawn at every rerun. It is also sent to the API at every turn: as on Thursday of last week, the API has no memory.
- `st.write_stream` shows the answer word by word as it arrives, from an OpenAI call with `stream=True`.

**Keys.** Never write a key in an app's code: the code goes to GitHub, and a deployed app's code can be public. Locally, the app reads `.env` as your notebooks do. On Streamlit Cloud, keys go in the app's **secrets**, which the app reads with `st.secrets["OPENAI_API_KEY"]`. To try secrets locally, create `.streamlit/secrets.toml` next to this README:

```toml
OPENAI_API_KEY = "sk-..."
```

Both `.env` and `.streamlit/secrets.toml` must be in `.gitignore`.

## 6. Your RAG pipeline as an app

`apps/06_rag_app.py` puts Friday's pipeline behind a text box: it loads the AI Act from `data/`, chunks it with page and article metadata, embeds the chunks with MiniLM and stores them in Chroma, all inside a function decorated with `@st.cache_resource`, so that the indexing (a few seconds) runs once per session, not at every question. The sidebar exposes `k` and the "articles only" filter from Friday, so a user can see how they change the answer.

The answer is shown with its **sources**: each retrieved excerpt in an expander, with a check mark on the ones the answer cites. For an LLM app, showing the sources is the single most useful thing you can do for your users: it lets them check.

## 7. More than one page

When an app grows, split it into pages: put the main script in the project folder and one file per page in a `pages/` folder next to it. Streamlit adds a navigation menu in the sidebar automatically, ordered by file name:

```
my_app/
├── Home.py                 streamlit run Home.py
└── pages/
    ├── 1_Explore.py
    ├── 2_Predict.py
    └── 3_Ask.py
```

Values in `st.session_state` are shared between pages, and cached functions are shared too.

## 8. Deploying on Streamlit Community Cloud

Streamlit Community Cloud runs public apps for free from a GitHub repository.

1. Push your app to a **public GitHub repository**, with a `requirements.txt` listing every library it imports (`streamlit`, `pandas`, `scikit-learn`, `openai`...) and the files it reads (data, saved model), if they are small and allowed to be public.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account, click **Create app**, and choose the repository, the branch and the main file (for example `app/streamlit_app.py`).
3. In **Advanced settings → Secrets**, paste your keys in the same TOML format as `secrets.toml`. They are stored by Streamlit, not in your repository.
4. Click **Deploy**. After a few minutes, the app has a public URL that you can share.

[example screenshot of the Streamlit Community Cloud "Create app" form: repository, branch, main file path, and the Advanced settings with Secrets]

The free tier has limits: a modest amount of memory, and apps that go to sleep when nobody uses them. A big fine-tuned transformer may not fit: deploy the smaller model, or keep that demo local. And **a public app that calls the OpenAI API spends your money every time anyone uses it**: set a spending limit on the key, or keep the app private.

## 9. Testing an app without a browser

Clicking through an app after every change does not scale. `streamlit.testing.v1.AppTest` runs an app headless and lets your code play the user: set widget values, click buttons, and check what the page shows.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("apps/01_hello.py").run()
at.text_input[0].input("Ana").run()
assert not at.success                     # nothing is greeted before the click
at.button[0].click().run()
assert at.success[0].value == "Hello Ana!"
```

`python test_apps.py` runs all six apps this way and prints what each one showed. It is the kind of check a gatekeeper runs before merging (Tuesday's DEV to PROD workflow), and the Project 3 rubric rewards an app that runs from a fresh clone.

## 10. Publishing apps responsibly

- **Keys and costs.** Keys in secrets, never in code. A public LLM app needs a spending limit, and ideally a login or a usage limit per session.
- **Disclose the AI.** If users chat with an LLM, say so on the page (the EU AI Act's transparency rule from Friday). Say what the app is not: *not medical advice*, *not legal advice*.
- **Personal data.** Whatever users type may be personal data, and an LLM app sends it to a provider. Do not log it without a reason, and do not deploy apps over data you are not allowed to publish.
- **Errors and uncertainty.** Validate inputs, catch API errors with a clear message instead of a stack trace, and show uncertainty: a low-confidence warning, an "I don't know", the sources.

---

## Summary

| Idea | Code |
|---|---|
| Run an app | `streamlit run app.py` |
| Text and numbers | `st.title`, `st.write`, `st.markdown`, `st.metric` |
| Inputs | `st.text_input`, `st.selectbox`, `st.slider`, `st.button`, ... return their current value |
| The rerun model | the whole script reruns at every interaction; `st.button` is `True` only in the run after the click |
| Layout | `st.sidebar`, `st.columns`, `st.tabs`, `st.expander`, `st.form` |
| Remember values | `st.session_state` |
| Cache data | `@st.cache_data` (a copy per caller) |
| Cache models, clients, indexes | `@st.cache_resource` (one shared object) |
| Charts and tables | `st.bar_chart`, `st.line_chart`, `st.pyplot(fig)`, `st.dataframe` |
| Chat | `st.chat_input`, `st.chat_message`, `st.write_stream` |
| Keys | `st.secrets["OPENAI_API_KEY"]`, from `.streamlit/secrets.toml` or the Cloud's secrets |
| Several pages | a `pages/` folder |
| Deploy | a public GitHub repo + share.streamlit.io |
| Test | `streamlit.testing.v1.AppTest` |

In the lab, you build a three-page dashboard for the Sakila DVD-rental database that you queried in Week 2: a home page, an exploration page, and a page that finds similar films with sentence embeddings. On Thursday and in Project 3, you will build the app for your own product.

## Appendix

### A.1 Faster partial reruns

`@st.fragment` reruns only one function when its widgets change, instead of the whole script: useful when one corner of a big dashboard changes often.

### A.2 Other frameworks

**Gradio** is popular for quick demos of machine learning models (it powers many Hugging Face Spaces), **Dash** for complex dashboards, and **FastAPI** for serving a model as an API that other programs call. Streamlit sits in between: fast to write, and flexible enough for a real internal tool.
