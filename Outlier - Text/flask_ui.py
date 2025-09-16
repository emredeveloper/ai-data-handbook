import os
import sys
import subprocess
from flask import Flask, render_template_string


INDEX_HTML = """
<!doctype html>
<title>Outlier Text Pipeline</title>
<h2>Outlier Text Pipeline</h2>
<p>Embedding model: <b>text-embedding-embeddinggemma-300m</b> (LM Studio)</p>
<form action="/run" method="post">
  <button type="submit">Çalıştır (HEADLESS)</button>
  <p style="font-size:0.9em;color:#666">Çıktı görselleri 'outputs' klasörüne kaydedilir.</p>
  <p style="font-size:0.9em;color:#666">Log çıktısının son kısmı aşağıda gösterilir.</p>
  <hr/>
  {% if msg %}<pre>{{ msg }}</pre>{% endif %}
</form>
"""


def launch_pipeline_headless() -> str:
    env = os.environ.copy()
    env["HEADLESS"] = "1"
    python_exe = sys.executable or "python"
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    try:
        proc = subprocess.run([python_exe, app_path], cwd=os.path.dirname(__file__), env=env, capture_output=True, text=True, timeout=3600)
        output = proc.stdout + "\n" + proc.stderr
        return output[-4000:]
    except Exception as e:
        return f"Çalıştırma hatası: {e}"


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(INDEX_HTML, msg=None)

    @app.route("/run", methods=["POST"])
    def run_pipeline():
        msg = launch_pipeline_headless()
        return render_template_string(INDEX_HTML, msg=msg)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)


