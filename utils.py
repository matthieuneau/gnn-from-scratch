def count_parameters(model):
    print(f"{'Layer':<40} {'# Parameters':>15}")
    print("=" * 60)

    for name, param in model.named_parameters():
        print(name)
        print(param)
        exit()
