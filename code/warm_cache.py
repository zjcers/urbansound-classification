import multiprocessing
from common import load_data, FOLD

def selector(fold):
    def real_selector(row):
        return row[FOLD] == fold
    return real_selector

def perform_fold_load(fold):
    selector_func = selector(str(fold))
    load_data(fold_selector=selector_func)

if __name__ == "__main__":
    with multiprocessing.Pool(5) as pool:
        results = []
        for fold in range(1, 11):
            # results.append(pool.apply_async(perform_fold_load, (fold,), error_callback=print))
            perform_fold_load(fold)
        for result in results:
            result.wait()