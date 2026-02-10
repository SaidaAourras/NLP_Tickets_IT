import time

print("🚀 Test Kubernetes : démarrage")
time.sleep(2)

for i in range(3):
    print(f"Traitement batch {i+1}/3")
    time.sleep(1)

print("✅ Fin du job Kubernetes")
