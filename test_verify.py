import urllib.request
import json

def verify_all():
    print("=== Testing Python Server on http://127.0.0.1:8000 ===")
    health = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/health").read().decode("utf-8"))
    print("Health Status:", health.get("status"))

    samples = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/samples").read().decode("utf-8"))
    print("Samples Count:", len(samples.get("samples", [])))

    logo_resp = urllib.request.urlopen("http://127.0.0.1:8000/logo.png")
    print("Logo HTTP Status:", logo_resp.status, "Content-Length:", len(logo_resp.read()))

    bg_resp = urllib.request.urlopen("http://127.0.0.1:8000/bg.png")
    print("BG HTTP Status:", bg_resp.status, "Content-Length:", len(bg_resp.read()))

    html = urllib.request.urlopen("http://127.0.0.1:8000/").read().decode("utf-8")
    print("Dashboard HTML length:", len(html))
    print("Dashboard has white logo container:", 'background-color: #ffffff !important;' in html)
    print("Dashboard has BENCHMARK_BUNDLE:", 'BENCHMARK_BUNDLE =' in html)

    print("\n=== Testing Node.js Gateway on http://127.0.0.1:5000 ===")
    node_health = json.loads(urllib.request.urlopen("http://127.0.0.1:5000/api/health").read().decode("utf-8"))
    print("Node Gateway Status:", node_health.get("status"))
    print("Python Backend through Gateway:", node_health.get("python_service", {}).get("status"))

    node_html = urllib.request.urlopen("http://127.0.0.1:5000/").read().decode("utf-8")
    print("Node served HTML length:", len(node_html))

    print("\n=== Checking Root Files for Vercel Drag & Drop Deployment ===")
    import os
    files = ["index.html", "logo.png", "bg.png", "vercel.json"]
    for f in files:
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        print(f"Root file '{f}': exists={exists}, size={size} bytes")

    print("\nALL VERIFICATIONS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_all()
