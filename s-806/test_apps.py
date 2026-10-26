"""Run every app of this lesson without a browser, the way a test would, and report what happened.

    python test_apps.py

Steps 5 and 6 are skipped if no OpenAI key is found in .env.
"""

import os

from dotenv import load_dotenv
from streamlit.testing.v1 import AppTest

load_dotenv()
HAS_KEY = bool(os.environ.get("OPENAI_API_KEY"))


def check(name, at):
    status = "OK" if not at.exception else f"ERROR {at.exception[0].message}"
    print(f"{name:28s} {status}")
    return at


at = check("01 hello", AppTest.from_file("apps/01_hello.py").run())
at.text_input[0].input("Ana").run()
assert not at.success, "nothing is shown before the click"
at.button[0].click().run()
print("   after the click:", at.success[0].value)

at = check("02 widgets and layout", AppTest.from_file("apps/02_widgets_and_layout.py").run())
at.button(key="counter").click().run()
at.button(key="counter").click().run()
print("   session state:", [m.value for m in at.markdown if "clicked" in m.value][0])

at = check("03 data and charts", AppTest.from_file("apps/03_data_and_charts.py").run())
at.slider[0].set_value(4).run()
print("   4+ service calls:", [(m.label, m.value) for m in at.metric])

if not os.path.exists("models/churn_pipeline.joblib"):
    os.system("python train_churn_model.py")
at = check("04 model app", AppTest.from_file("apps/04_model_app.py").run())
at.selectbox[0].set_value("yes")
at.number_input[0].set_value(5)
at.button[0].click().run()
print("   international plan, 5 service calls:", at.metric[0].value)

if HAS_KEY:
    at = check("05 chatbot", AppTest.from_file("apps/05_chatbot.py", default_timeout=60).run())
    at.chat_input[0].set_value("My card was blocked after three wrong PINs. What can I do?").run()
    print("   answer:", at.session_state.messages[-1]["content"][:150])

    at = check("06 RAG app", AppTest.from_file("apps/06_rag_app.py", default_timeout=180).run())
    at.text_input[0].input("How much can a company be fined for using a banned AI practice?").run()
    print("   answer:", at.markdown[0].value[:200])
else:
    print("05 chatbot and 06 RAG app   skipped: no OPENAI_API_KEY in .env")
