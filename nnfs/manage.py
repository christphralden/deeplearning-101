import argparse
from data import get_dataset, show_images

parser = argparse.ArgumentParser(description='NNFS')

subparsers = parser.add_subparsers(dest='command')

dataset_parser = subparsers.add_parser('dataset', help='Dataset-related operations')
dataset_parser.add_argument('--show', action='store_true', help='Show dataset statistics')
dataset_parser.add_argument('--with-image', action='store_true', help='Display images from the dataset')
dataset_parser.add_argument('--alphanumeric', action='store_true', help='Adds the alphanumeric dataset')

args = parser.parse_args()

if args.command == 'dataset':
    print("Loading dataset...")
    
    if args.alphanumeric:
        x_train, x_test, y_train, y_test = get_dataset(
            select_class="ALPHANUMERIC"
        )
    else:
        x_train, x_test, y_train, y_test = get_dataset()

    if args.show:
        print(f"Dataset loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} testing samples.")
    
    if args.with_image:
        print("Displaying a few training images...")
        show_images(x_train.reshape(-1, 50, 50), num_row=2, num_col=5)
else:
    parser.print_help()
