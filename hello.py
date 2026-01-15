from pathlib import Path

def tester_encodages_csv():
    """Teste différents encodages pour lire le CSV Cars Datasets 2025"""
    csv_path = Path("Cars Datasets 2025.csv")
    
    if not csv_path.exists():
        print(f"❌ Fichier introuvable: {csv_path}")
        return
    
    encodages = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1", "windows-1252"]
    
    for enc in encodages:
        try:
            with open(csv_path, "r", encoding=enc) as f:
                # Lire les 3 premières lignes
                lines = [f.readline() for _ in range(3)]
            
            print(f"✅ {enc:20} → SUCCÈS")
            print(f"   Première ligne: {lines[0][:80]}...")
            print(f"   Deuxième ligne: {lines[1][:80]}...")
            print()
            
        except (UnicodeDecodeError, LookupError) as e:
            print(f"❌ {enc:20} → ÉCHEC: {type(e).__name__}")

if __name__ == "__main__":
    tester_encodages_csv()
