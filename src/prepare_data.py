"""
Préparation du dataset ASRS pour fine-tuning AeroLLM.
Source : elihoole/asrs-aviation-reports (NASA ASRS)
Format cible : instruction/input/output (Alpaca format)
"""
import json
import os
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()


def load_asrs_dataset():
    """Charge le dataset ASRS depuis HuggingFace."""
    console.print("[bold blue]📥 Chargement dataset ASRS...[/bold blue]")

    dataset = load_dataset(
        "elihoole/asrs-aviation-reports",
        split="train"
    )

    console.print(f"✅ {len(dataset)} rapports chargés")
    console.print(f"Colonnes disponibles : {len(dataset.column_names)}")
    return dataset


def format_to_alpaca(dataset, max_samples: int = 2000) -> list:
    """
    Convertit les rapports ASRS au format Alpaca.
    
    Le LLM apprend à analyser un rapport d'incident et produire :
    - Causes identifiées
    - Facteurs contributifs
    - Actions correctives recommandées
    """
    console.print("[bold blue]🔄 Formatage au format Alpaca...[/bold blue]")

    formatted = []
    skipped = 0

    for item in track(dataset, description="Formatage..."):
        if len(formatted) >= max_samples:
            break

        # Colonnes clés du dataset ASRS
        narrative = item.get("Report 1_Narrative", "") or ""
        synopsis = item.get("Report 1.2_Synopsis", "") or ""
        anomaly = item.get("Events_Anomaly", "") or ""
        contributing = item.get("Assessments_Contributing Factors / Situations", "") or ""
        primary_problem = item.get("Assessments.1_Primary Problem", "") or ""
        aircraft = item.get("Aircraft 1.2_Make Model Name", "") or ""
        flight_phase = item.get("Aircraft 1.9_Flight Phase", "") or ""
        human_factors = item.get("Person 1.7_Human Factors", "") or ""
        result = item.get("Events.5_Result", "") or ""

        # Filtre : narrative trop courte
        if len(narrative) < 150:
            skipped += 1
            continue

        # Construit l'output structuré à partir des vraies données ASRS
        output_parts = ["**Analyse de l'incident aéronautique**\n"]

        if synopsis:
            output_parts.append(f"**Résumé :** {synopsis}\n")

        if anomaly:
            output_parts.append(f"**Anomalie détectée :** {anomaly}\n")

        if contributing or primary_problem:
            output_parts.append("**Facteurs contributifs :**")
            if contributing:
                output_parts.append(f"- {contributing}")
            if primary_problem:
                output_parts.append(f"- Problème principal : {primary_problem}")
            if human_factors:
                output_parts.append(f"- Facteurs humains : {human_factors}")
            output_parts.append("")

        if result:
            output_parts.append(f"**Actions prises :** {result}\n")

        output_parts.append(
            "**Recommandations :**\n"
            "- Revue des procédures opérationnelles standard (SOP)\n"
            "- Debriefing d'équipage recommandé\n"
            "- Rapport transmis aux autorités compétentes (FAA/EASA)"
        )

        entry = {
            "instruction": (
                "Tu es un expert en sécurité aéronautique certifié FAA/EASA. "
                "Analyse ce rapport d'incident ASRS et fournis une analyse structurée : "
                "causes identifiées, facteurs contributifs, et recommandations."
            ),
            "input": (
                f"Aéronef : {aircraft}\n"
                f"Phase de vol : {flight_phase}\n"
                f"Rapport : {narrative[:800]}"
            ),
            "output": "\n".join(output_parts)
        }
        formatted.append(entry)

    console.print(f"✅ {len(formatted)} exemples formatés ({skipped} ignorés)")
    return formatted


def save_dataset(formatted: list, output_path: str = "data/processed/train.json"):
    """Sauvegarde le dataset formaté."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    console.print(f"💾 Dataset sauvegardé : {output_path} ({size_mb:.1f} MB)")


def preview_dataset(formatted: list, n: int = 2):
    """Affiche un aperçu du dataset."""
    console.print(f"\n[bold]📋 Aperçu ({n} exemples) :[/bold]")
    for i, item in enumerate(formatted[:n]):
        console.print(f"\n[cyan]--- Exemple {i+1} ---[/cyan]")
        console.print(f"[yellow]Input:[/yellow]\n{item['input'][:300]}")
        console.print(f"[yellow]Output:[/yellow]\n{item['output'][:300]}")


if __name__ == "__main__":
    console.print("[bold]🛩️ AeroLLM — Préparation Dataset ASRS[/bold]\n")

    dataset = load_asrs_dataset()
    formatted = format_to_alpaca(dataset, max_samples=2000)
    preview_dataset(formatted)
    save_dataset(formatted)

    console.print("\n[bold green]✅ Dataset prêt pour le fine-tuning ![/bold green]")
    console.print(f"[dim]→ {len(formatted)} exemples dans data/processed/train.json[/dim]")