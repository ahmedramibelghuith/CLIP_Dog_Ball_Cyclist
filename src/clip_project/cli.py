"""Interface CLI ; l'aide reste disponible sans installer TensorFlow."""
import argparse


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Une valeur strictement positive est requise.")
    return number


def parser():
    result = argparse.ArgumentParser(description="Pipeline CLIP : entraînement, indexation et recherche.")
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Entraîner les deux classifieurs puis CLIP")
    train.add_argument("--csv", required=True)
    train.add_argument("--image-root", required=True, help="Racine des chemins image_path du CSV")
    train.add_argument("--output", default="artifacts/run-001")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--classifier-epochs", type=positive, default=8)
    train.add_argument("--clip-epochs", type=positive, default=10)
    train.add_argument("--sequence-length", type=positive, default=32)
    train.add_argument("--batch-size", type=positive, default=32)
    index = commands.add_parser("index", help="Construire ou remplacer l'index de recherche")
    index.add_argument("--csv", required=True)
    index.add_argument("--image-root", required=True)
    search = commands.add_parser("search", help="Rechercher par texte ou image")
    query = search.add_mutually_exclusive_group(required=True)
    query.add_argument("--text")
    query.add_argument("--image")
    search.add_argument("--top-k", type=positive, default=5)
    evaluate = commands.add_parser("evaluate", help="Mesurer Recall@K sur le test du run")
    for command in (index, search, evaluate):
        command.add_argument("--run", required=True)
    for command in (index, evaluate):
        command.add_argument("--batch-size", type=positive, default=32)
    return result


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if args.command == "train" and args.batch_size < 2:
        command_parser.error("CLIP nécessite --batch-size >= 2.")
    try:
        if args.command == "train":
            from .pipeline import train
            train(args)
        else:
            from .retrieval import build_index, search, evaluate
            {"index": build_index, "search": search, "evaluate": evaluate}[args.command](args)
    except (FileNotFoundError, ValueError, ImportError) as error:
        command_parser.exit(1, f"Erreur : {error}\n")
