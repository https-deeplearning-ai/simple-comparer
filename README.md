Perfecto 👌 — dado que tu repositorio se llama **`simple-comparer`**, aquí tienes la versión ajustada de la descripción y el texto largo (para el README o la cabecera de GitHub):

---

### 🧩 **Simple Comparer – OpenAI Model Evaluator**

An interactive **FastAPI + Tailwind + Chart.js** web app to **compare OpenAI models side-by-side**.
It lets you send the same prompt to two different models, visualize responses, and automatically evaluate them using an LLM-based or heuristic judge.

**Key features:**

* 🔹 Real-time comparison of OpenAI models (GPT-4o, GPT-4.1, GPT-3.5, etc.)
* 🔹 Automatic evaluation with structured metrics (Clarity, Task Fit, Structure, Safety, Correctness)
* 🔹 Topic extraction and shared-topic visualization via **Venn diagrams**
* 🔹 Compact **radar**, **bar**, and **scatter** charts for insights
* 🔹 Built with **FastAPI**, **TailwindCSS**, **Jinja2**, and **Chart.js**
* 🔹 CSV export of full comparison history

**Ideal for:** quick model comparisons, LLM evaluation demos, and educational projects.

---

### ⚙️ Quick start

```bash
git clone https://github.com/yourname/simple-comparer.git
cd simple-comparer
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-proj-..." > .env
uvicorn main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) and start comparing models.

---
