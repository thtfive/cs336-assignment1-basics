from cs336_basics.train_bpe import run_train_bpe_on_tiny_stories


# Example:
#   python -m cProfile -o profile_result.prof -m cs336_basics.profile_train_bpe
# Then analyze:
#   python -m pstats profile_result.prof -s tottime -p 30
# Or visualize:
#   snakeviz profile_result.prof
if __name__ == "__main__":
    run_train_bpe_on_tiny_stories()