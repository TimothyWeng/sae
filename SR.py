from sae import Sae
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from collections import Counter
from pysr import PySRRegressor
import argparse
import gc
import os


# Helper functions
def extract_flattened_acts_indices(layer_acts):
    acts, indices = [], []
    for enc_out in layer_acts:
        acts_list = enc_out['top_acts'].tolist()
        indices_list = enc_out['top_indices'].tolist()
        for act, ind in zip(acts_list, indices_list):
            acts.extend(act)
            indices.extend(ind)
    return acts, indices


def filter_by_neuron(front_acts, front_indices, back_acts, back_indices, neuron):
    filt_front_acts, filt_front_indices = [], []
    filt_back_acts, filt_back_indices = [], []

    for i, (b_acts, b_indices) in enumerate(zip(back_acts, back_indices)):
        if neuron in b_indices:
            filt_back_acts.append(b_acts)
            filt_back_indices.append(b_indices)
            filt_front_acts.append(front_acts[i])
            filt_front_indices.append(front_indices[i])

    return filt_front_acts, filt_front_indices, filt_back_acts, filt_back_indices


def update_acts(acts, indices):
    unified_indices = sorted({ind for ind_list in indices for ind in ind_list})
    updated_acts = [
        [dict(zip(idx, act)).get(neuron, 0) for neuron in unified_indices]
        for act, idx in zip(acts, indices)
    ]
    return updated_acts, unified_indices


def extract_Y(back_acts, back_indices, neuron):
    return [
        act[indices.index(neuron)]
        for act, indices in zip(back_acts, back_indices)
        if neuron in indices
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Symbolic regression on specific neurons"
    )
    parser.add_argument(
        "--neuron_start", type=int, required=True, help="Starting neuron index"
    )
    parser.add_argument(
        "--neuron_end", type=int, required=True, help="Ending neuron index"
    )
    args = parser.parse_args()

    # Replace this with argument-based selection
    neuron_start = args.neuron_start
    neuron_end = args.neuron_end

    # Set the specific layers to load (e.g., layer 5 as front, layer 6 as back)
    front_layer = 1
    back_layer = 2

    # Determine if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SAEs for the specified layers and move them to the correct device
    saes = {
        "front": Sae.load_from_hub(
            "EleutherAI/sae-pythia-70m-deduped-32k", hookpoint=f"layers.{front_layer}"
        ).to(device),  # Move SAE to device
        "back": Sae.load_from_hub(
            "EleutherAI/sae-pythia-70m-deduped-32k", hookpoint=f"layers.{back_layer}"
        ).to(device),  # Move SAE to device
    }

    # Load dataset and tokenizer
    dataset = load_dataset(
        "EleutherAI/the_pile_deduplicated", split="train", streaming=True
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped").to(
        device
    )  # Move model to device

    # Parameters
    batch_size = 4
    max_batches = 1000
    max_samples = batch_size * max_batches
    best_score = 0
    best_sym_model = None
    best_target_indice = 0

    # Initialize trackers and activations
    common_indices = Counter()  # Track activations in the back layer
    activations = {"front": [], "back": []}  # Store latent activations
    batch_texts = []
    processed_samples = 0

    model.eval()

    # Use `torch.no_grad()` for the entire loop to prevent gradient tracking and reduce memory usage
    with torch.no_grad():
        # Process dataset
        for sample in tqdm(dataset, total=max_samples, desc="Processing samples"):
            batch_texts.append(sample["text"])
            processed_samples += 1

            if len(batch_texts) == batch_size:
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)  # Move inputs to GPU

                outputs = model(**inputs, output_hidden_states=True)

                # Collect latent activations for front and back layers
                for layer_name, (sae, hidden_state) in zip(
                    saes.keys(), zip(saes.values(), outputs.hidden_states)
                ):
                    hidden_state = hidden_state.to(
                        device
                    )  # Ensure hidden states are on the GPU
                    latent_acts = sae.encode(hidden_state)  # Encoding happens on GPU

                    # Move tensors to CPU
                    top_acts_cpu = latent_acts.top_acts.cpu()  # Move activations to CPU
                    top_indices_cpu = (
                        latent_acts.top_indices.cpu()
                    )  # Move indices to CPU

                    # Optionally store or discard activations (depending on your need)
                    # You can reduce the memory usage here by only keeping necessary data.
                    # For example, clear activations every few batches or write to disk
                    activations[layer_name].append(
                        {"top_acts": top_acts_cpu, "top_indices": top_indices_cpu}
                    )

                    if layer_name == "back":
                        # Update common_indices with the CPU-based indices
                        common_indices.update(top_indices_cpu.flatten().tolist())

                # Free GPU memory after each batch
                del outputs, inputs, hidden_state, latent_acts  # Free memory explicitly
                torch.cuda.empty_cache()  # Clear cached GPU memory
                gc.collect()  # Run garbage collection for CPU memory

                batch_texts = []  # Clear batch

            if processed_samples >= max_samples:
                break

    # Extract and process latent activations
    front_acts, front_indices = extract_flattened_acts_indices(activations["front"])
    back_acts, back_indices = extract_flattened_acts_indices(activations["back"])

    # Get most frequent neuron in back layer
    for target_neuron in range(neuron_start, neuron_end):
        print(target_neuron)
        # Filter activations based on target neuron
        f_front_acts, f_front_indices, f_back_acts, f_back_indices = filter_by_neuron(
            front_acts, front_indices, back_acts, back_indices, target_neuron
        )

        # Update front layer activations
        X, unified_indices = update_acts(f_front_acts, f_front_indices)
        Y = extract_Y(f_back_acts, f_back_indices, target_neuron)

        print(len(X), len(Y))
        if len(X) == 0:
            continue
        print(f"feature dimemsion: {len(X[0])}")

        # Create the directory if it doesn't exist
        os.makedirs(f"equations/{target_neuron}", exist_ok=True)

        # Train symbolic regression model
        sym_model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/", "cond", "logical_or", "logical_and"],
            unary_operators=["square", "cos", "sin", "exp", "log", "inv(x) = 1/x"],
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            batching=True,
            equation_file=f"equations/{target_neuron}/equation.csv",  # Full path with file name
        )
        sym_model.fit(X, Y)

        equations_file = f"equations/{target_neuron}/equation.txt"
        with open(equations_file, "w") as file:
            file.write(sym_model.equations_.to_string())
            file.write("\n\n")
            file.write(f"model equation =\n{sym_model.get_best()['equation']}")
            file.write("\n\n")
            file.write(f"model score =\n{sym_model.get_best()['score']}")

        if len(X) >= 100 and sym_model.get_best()["score"] > best_score:
            best_score = sym_model.get_best()["score"]
            best_sym_model = sym_model
            best_target_indice = target_neuron

    output_file = f"equations/best{neuron_start}-{neuron_end}.txt"
    if best_sym_model is not None:
        with open(output_file, "w") as file:
            file.write(best_sym_model.equations_.to_string())
            file.write("\n\n")
            file.write(f"target neuron = {best_target_indice}")
            file.write("\n\n")
            file.write(f"model equation =\n{best_sym_model.get_best()['equation']}")
            file.write("\n\n")
            file.write(f"model score =\n{best_sym_model.get_best()['score']}")


# Entry point
if __name__ == "__main__":
    main()
