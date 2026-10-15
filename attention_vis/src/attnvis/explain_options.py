"""Shared explanation CLI options."""
def add_explain_options(parser):
    parser.add_argument("--method", choices=("attention","rollout","gradient_rollout"), default="attention")
    parser.add_argument("--action-step", type=int, default=0, help="Action token index within chunk")
    parser.add_argument("--action-dim", type=int, default=0, help="Normalized action dimension for gradient_rollout")
    parser.add_argument("--target-sign", type=int, choices=(-1,1), default=1)
    parser.add_argument("--seed", type=int, default=0, help="Initial action noise seed for rollout methods")
