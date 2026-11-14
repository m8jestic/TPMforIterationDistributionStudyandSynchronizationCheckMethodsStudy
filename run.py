import hashlib
import math
import os

import pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numpy.random as rnd


def createKey(n: int, Q: float):
    AliceKey = rnd.randint(0, 2, size=(n))
    error_line = np.array([1 if rnd.random() < Q else 0 for _ in range(n)])
    BobKey = AliceKey ^ error_line
    return AliceKey, BobKey


def list_to_matrix(input_list, k, n):
    if len(input_list) < k * n:
        while len(input_list) != k * n:
            input_list.append(0)
    matrix = np.array(input_list).reshape(k, n)
    return matrix


class TreeParityMachine:
    def __init__(self, K, N, L, weights):
        self.K = K
        self.N = N
        self.L = L
        self.weights = weights

    def update(self, X, tau_Partner):
        tau = self.tau
        sigma = self.sigma
        weights = self.weights
        L = self.L
        if tau == tau_Partner:
            self.hebbian(weights, X, sigma, tau, tau_Partner, L)

    def compute_output(self, X):
        X = X.reshape([self.K, self.N])
        sigma = np.where(np.sum(X * self.weights, axis=1) > 0, 1, -1)
        tau = np.prod(sigma)
        self.X = X
        self.sigma = sigma
        self.tau = tau
        return tau

    def hebbian(self, weights, X, sigma, tau_A, tau_B, l):
        mask = (sigma == tau_A).astype(weights.dtype).reshape(-1, 1)
        weights += X * tau_A * mask
        np.clip(weights, -l, l, out=weights)


def binary_to_decimal(binary_array, rangeBits):
    weightList = []
    minus = 0
    for k in range(0, rangeBits):
        minus += 2 ** k
    minus = minus // 2
    minus += 1
    for i in range(0, len(binary_array), rangeBits):
        torture = ''
        for j in range(0, rangeBits):
            torture += str(binary_array[i + j])
        a = int(torture, 2)
        a -= minus
        weightList.append(a)
    return weightList


def compute_hash(weights):
    W = np.asarray(weights, dtype=int)
    return hashlib.sha256(W.tobytes()).hexdigest()


def get_parameter_sets_for_key_length(target_key_length=8192, bits_min=1, bits_max=13):
    parameter_sets = []
    for Bits in range(bits_min, bits_max + 1):
        if target_key_length % Bits != 0:
            continue
        product_kn = target_key_length // Bits
        for K in range(1, product_kn + 1):
            if product_kn % K == 0:
                N = product_kn // K
                if K <= N:
                    parameter_sets.append([K, N, Bits])
    return parameter_sets


def _measure_chunk(args):
    (K, N, Bits, kStart, kLimit, TCounterLimit, sync_check,
     measurements_chunk, max_iterations, Q, seed) = args

    rnd.seed(seed)

    n_local = K * N * Bits
    L_local = 2 ** Bits
    L_local = L_local // 2
    M = L_local

    failures = 0
    sync_iterations_sum = 0
    disclosed_information_sum = 0.0
    fact_sync_iterations_sum = 0
    truepositive_sum = 0
    falsepositive_sum = 0
    total_len_messages = 0
    total_count_messages = 0

    for _ in range(measurements_chunk):
        keySlice_A, keySlice_B = createKey(n_local, Q)

        weights_A = binary_to_decimal(keySlice_A, Bits)
        weights_B = binary_to_decimal(keySlice_B, Bits)

        arrayWeights_A = list_to_matrix(weights_A, K, N)
        arrayWeights_B = list_to_matrix(weights_B, K, N)

        TPM_A = TreeParityMachine(K, N, L_local, arrayWeights_A)
        TPM_B = TreeParityMachine(K, N, L_local, arrayWeights_B)

        iter_ = 0
        iterations = 0
        TCounter = 0
        kCounter = 0
        hashCounter = 3
        TCounterStarted = False
        factSyncIter = 0
        factSyncFlag = False

        while iter_ < max_iterations:
            iter_ += 1
            iterations += 1


            X = np.random.randint(-1, 2, [K, N])

            out_A = TPM_A.compute_output(X)
            out_B = TPM_B.compute_output(X)

            if out_A == out_B:
                TPM_A.update(X, out_B)
                TPM_B.update(X, out_A)

                weights_equal = np.array_equal(
                    TPM_A.weights.flatten(), TPM_B.weights.flatten()
                )

                if weights_equal and not factSyncFlag:
                    factSyncFlag = True
                    factSyncIter = iter_

                if sync_check == "none":
                    if weights_equal:
                        fact_sync_iterations_sum += factSyncIter
                        sync_iterations_sum += iterations
                        disclosed_information_sum += 2 * iterations
                        truepositive_sum += 1
                        break

                elif sync_check == "counter":
                    TCounter += 1

                    if TCounter == TCounterLimit:
                        if weights_equal:
                            fact_sync_iterations_sum += factSyncIter
                            sync_iterations_sum += iterations
                            disclosed_information_sum += 2 * iterations
                            truepositive_sum += 1
                        else:
                            falsepositive_sum += 1
                        break

                elif sync_check == "hash":
                    if kCounter == 3 * kLimit:
                        hash_A = compute_hash(TPM_A.weights)
                        hash_B = compute_hash(TPM_B.weights)

                        hash_A_int = int(hash_A, 16)
                        hash_B_int = int(hash_B, 16)

                        bit_mask = (1 << kLimit) - 1
                        abs_P_A = f"{(hash_A_int & bit_mask):0{kLimit}b}"
                        abs_P_B = f"{(hash_B_int & bit_mask):0{kLimit}b}"

                        total_count_messages += 1
                        total_len_messages += kLimit

                        if abs_P_A == abs_P_B:
                            if weights_equal:
                                fact_sync_iterations_sum += factSyncIter
                                sync_iterations_sum += iterations
                                disclosed_information_sum += 2 * iterations
                                truepositive_sum += 1
                            else:
                                falsepositive_sum += 1
                            break
                        else:
                            kCounter = 0
                    else:
                        kCounter += 1

                elif sync_check == "hybrid":
                    TCounter += 1
                    if ((not TCounterStarted and TCounter >= TCounterLimit) or
                            (TCounterStarted and
                             TCounter >= (TCounterLimit // 5))):
                        TCounterStarted = True

                        if hashCounter >= 3 * kStart:
                            if kCounter == 0:
                                prog_k = 1
                            elif kCounter == 1:
                                prog_k = 1
                            elif kCounter == 2:
                                prog_k = 2
                            else:
                                prog_k = 3

                            hash_A = compute_hash(TPM_A.weights)
                            hash_B = compute_hash(TPM_B.weights)

                            hash_A_int = int(hash_A, 16)
                            hash_B_int = int(hash_B, 16)

                            bit_mask = (1 << prog_k) - 1
                            abs_P_A = f"{(hash_A_int & bit_mask):0{prog_k}b}"
                            abs_P_B = f"{(hash_B_int & bit_mask):0{prog_k}b}"

                            total_count_messages += 1
                            total_len_messages += prog_k

                            if abs_P_A == abs_P_B:
                                kCounter += 1
                            else:
                                kCounter = 0
                                hashCounter = 0

                            if abs_P_A == abs_P_B and kCounter >= kLimit:
                                if weights_equal:
                                    fact_sync_iterations_sum += factSyncIter
                                    sync_iterations_sum += iterations
                                    disclosed_information_sum += 2 * iterations
                                    truepositive_sum += 1
                                else:
                                    falsepositive_sum += 1
                                break
                        else:
                            hashCounter += 1

            elif sync_check == "counter":
                TCounter = 0
            elif sync_check == "hybrid":
                kCounter = 0
                TCounter = 0

            if iter_ == max_iterations:
                failures += 1

    return (
        failures,
        sync_iterations_sum,
        disclosed_information_sum,
        fact_sync_iterations_sum,
        truepositive_sum,
        falsepositive_sum,
        total_count_messages,
        total_len_messages,
        measurements_chunk,
    )


def measure_single_parameter_set_parallel(K, N, Bits, kStart=1, kLimit=5,
                                           TCounterLimit=24, measurements=10000,
                                           max_iterations=10000, Q=0.05,
                                           n_workers=None, verbose=True,
                                           sync_check="none"):
    if sync_check not in ("none", "counter", "hash", "hybrid"):
        raise ValueError(
            'sync_check должен иметь значение "none", "counter", "hash" или "hybrid"'
        )
    if sync_check == "counter" and TCounterLimit < 1:
        raise ValueError("TCounterLimit должен быть положительным")
    if sync_check == "hash" and not 1 <= kLimit <= 256:
        raise ValueError("Для hash-процедуры kLimit должен быть от 1 до 256")
    if sync_check == "hybrid":
        if kStart < 1:
            raise ValueError("Для гибридной процедуры kStart должен быть >= 1")
        if kLimit < 1:
            raise ValueError("Для гибридной процедуры kLimit должен быть >= 1")
        if TCounterLimit < 1:
            raise ValueError("Для гибридной процедуры TCounterLimit должен быть >= 1")
    if measurements < 1:
        raise ValueError("measurements должен быть не меньше 1")
    if max_iterations < 1:
        raise ValueError("max_iterations должен быть не меньше 1")

    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, measurements))

    base = measurements // n_workers
    remainder = measurements % n_workers
    chunks = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    chunks = [c for c in chunks if c > 0]

    seeds = [int.from_bytes(os.urandom(4), "little") for _ in chunks]

    tasks = [
        (K, N, Bits, kStart, kLimit, TCounterLimit, sync_check,
         chunk, max_iterations, Q, seed)
        for chunk, seed in zip(chunks, seeds)
    ]

    total_failures = 0
    total_sync_iterations = 0
    total_disclosed_information = 0.0
    total_fact_sync_iterations = 0
    total_truepositive = 0
    total_falsepositive = 0
    total_count_messages = 0
    total_len_messages = 0
    total_measurements = 0

    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futures = [ex.submit(_measure_chunk, task) for task in tasks]
        for future in as_completed(futures):
            (failures, sync_it, disc_info, fact_sync_it,
             truepositive, falsepositive, count_messages, len_messages,
             m) = future.result()
            total_failures += failures
            total_sync_iterations += sync_it
            total_disclosed_information += disc_info
            total_fact_sync_iterations += fact_sync_it
            total_truepositive += truepositive
            total_falsepositive += falsepositive
            total_count_messages += count_messages
            total_len_messages += len_messages
            total_measurements += m

    fer_average = total_failures / total_measurements
    decided_trials = total_measurements - total_failures

    if total_truepositive > 0:
        fact_sync_iterations_average = (
            total_fact_sync_iterations / total_truepositive
        )
        sync_iterations_average = total_sync_iterations / total_truepositive
        disclosed_information_average = (
            total_disclosed_information / total_truepositive
        )
    else:
        fact_sync_iterations_average = 0
        sync_iterations_average = 0
        disclosed_information_average = 0

    if decided_trials > 0:
        truepositive_average = total_truepositive / decided_trials
        falsepositive_average = total_falsepositive / decided_trials
    else:
        truepositive_average = 0
        falsepositive_average = 0

    if sync_check in ("hash", "hybrid") and total_truepositive > 0:
        total_count_messages_avg = total_count_messages / total_truepositive
        total_len_messages_avg = total_len_messages / total_truepositive
    else:
        total_count_messages_avg = 0
        total_len_messages_avg = 0

    if verbose:
        result = [
            K, N, Bits, fer_average,
            fact_sync_iterations_average,
            sync_iterations_average,
            disclosed_information_average,
        ]
        print("K", "N", "bits", "FER", "fact_sync_iterations_average",
              "sync_iterations_average", "disclosed_information_average")
        if sync_check in ("counter", "hash", "hybrid"):
            result.extend([truepositive_average, falsepositive_average])
        if sync_check in ("hash", "hybrid"):
            result.extend([total_count_messages_avg, total_len_messages_avg])
        print(*result)

    return {
        "K": K,
        "N": N,
        "Bits": Bits,
        "sync_check": sync_check,
        "fer_average": fer_average,
        "fact_sync_iterations_average": fact_sync_iterations_average,
        "sync_iterations_average": sync_iterations_average,
        "disclosed_information_average": disclosed_information_average,
        "truepositive_average": truepositive_average,
        "falsepositive_average": falsepositive_average,
        "totalCountMessages_avg": total_count_messages_avg,
        "totalLenMessages_avg": total_len_messages_avg,
        "n_workers": len(tasks),
    }


def _measure_sync_iteration_chunk(args):
    K, N, Bits, measurements_chunk, max_iterations, Q, seed = args
    rnd.seed(seed)

    n_local = K * N * Bits
    L_local = (2 ** Bits) // 2
    M = L_local

    fact_sync_iterations = []

    for _ in range(measurements_chunk):
        keySlice_A, keySlice_B = createKey(n_local, Q)

        weights_A = binary_to_decimal(keySlice_A, Bits)
        weights_B = binary_to_decimal(keySlice_B, Bits)

        arrayWeights_A = list_to_matrix(weights_A, K, N)
        arrayWeights_B = list_to_matrix(weights_B, K, N)

        TPM_A = TreeParityMachine(K, N, L_local, arrayWeights_A)
        TPM_B = TreeParityMachine(K, N, L_local, arrayWeights_B)

        factSyncIter = None

        for iter_ in range(1, max_iterations + 1):


            X = np.random.randint(-1, 2, [K, N])

            out_A = TPM_A.compute_output(X)
            out_B = TPM_B.compute_output(X)

            if out_A == out_B:
                TPM_A.update(X, out_B)
                TPM_B.update(X, out_A)

                weights_equal = np.array_equal(
                    TPM_A.weights.flatten(), TPM_B.weights.flatten()
                )

                if weights_equal:
                    factSyncIter = iter_
                    break

        fact_sync_iterations.append(factSyncIter)

    return fact_sync_iterations


def measure_sync_iteration_distribution_parallel(
        K, N, Bits, measurements=10000, Q=0.05,
        n_workers=None, output_file=None, verbose=True,
        max_iterations=None):
    if K < 1 or N < 1 or Bits < 1:
        raise ValueError("K, N и Bits должны быть положительными")
    if measurements < 1:
        raise ValueError("measurements должен быть не меньше 1")
    if not 0.0 <= Q <= 1.0:
        raise ValueError("Q должен находиться в диапазоне от 0 до 1")

    key_length = K * N * Bits

    if max_iterations is None:
        max_iterations = key_length

    if not 1 <= max_iterations <= key_length:
        raise ValueError(
            f"max_iterations должен находиться в диапазоне от 1 до {key_length}"
        )

    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, measurements))

    base = measurements // n_workers
    remainder = measurements % n_workers
    chunks = [base + (1 if i < remainder else 0)
              for i in range(n_workers)]
    chunks = [chunk for chunk in chunks if chunk > 0]

    seeds = [int.from_bytes(os.urandom(4), "little") for _ in chunks]
    tasks = [
        (K, N, Bits, chunk, max_iterations, Q, seed)
        for chunk, seed in zip(chunks, seeds)
    ]

    all_fact_sync_iterations = []

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(_measure_sync_iteration_chunk, task)
                   for task in tasks]
        for future in futures:
            all_fact_sync_iterations.extend(future.result())

    if len(all_fact_sync_iterations) != measurements:
        raise RuntimeError(
            "Число сохранённых результатов не совпадает с числом репликаций"
        )

    if output_file is None:
        output_file = f"resultK={K},N={N},bits={Bits}.xlsx"

    results_df = pd.DataFrame({
        "Итерация синхронизации": all_fact_sync_iterations,
    })
    results_df.to_excel(output_file, index=False)

    synchronized_count = sum(
        iteration is not None for iteration in all_fact_sync_iterations
    )
    failed_count = measurements - synchronized_count

    if verbose:
        print(
            f"[K={K}, N={N}, Bits={Bits}] "
            f"файл={os.path.abspath(output_file)}"
        )

    return {
        "K": K,
        "N": N,
        "Bits": Bits,
        "key_length": key_length,
        "max_iterations": max_iterations,
        "measurements": measurements,
        "synchronized_count": synchronized_count,
        "failed_count": failed_count,
        "iterations": all_fact_sync_iterations,
        "output_file": os.path.abspath(output_file),
        "n_workers": len(tasks),
    }


def read_positive_int(prompt, default=None):
    while True:
        value = input(prompt).strip()

        if value == "" and default is not None:
            return default

        try:
            value = int(value)
            if value > 0:
                return value
            print("Ошибка: необходимо ввести положительное целое число.")
        except ValueError:
            print("Ошибка: необходимо ввести целое число.")


def read_float(prompt, default=None, min_value=0.0, max_value=1.0):
    while True:
        value = input(prompt).strip()

        if value == "" and default is not None:
            return default

        try:
            value = float(value.replace(",", "."))
            if math.isfinite(value) and min_value <= value <= max_value:
                return value
            print(
                f"Ошибка: значение должно находиться "
                f"в диапазоне от {min_value} до {max_value}."
            )
        except ValueError:
            print("Ошибка: необходимо ввести вещественное число.")


def read_bounded_positive_int(prompt, max_value, default):
    while True:
        value = input(prompt).strip()

        if value == "":
            return default

        try:
            value = int(value)
            if 1 <= value <= max_value:
                return value
            print(f"Ошибка: значение должно быть от 1 до {max_value}.")
        except ValueError:
            print("Ошибка: необходимо ввести целое число.")


def main():
    print("=" * 85)
    print(
        "Программа для исследования методов синхронизации "
        "автоматов Кантера-Кинцеля с минимальным разглашением"
    )
    print("=" * 85)

    print("\nВыберите режим работы программы:")
    print("1. Исследование распределения итераций синхронизации")
    print("2. Исследование процедур проверки синхронизации")

    while True:
        mode = input("\nВведите номер режима (1 или 2): ").strip()
        if mode in ("1", "2"):
            mode = int(mode)
            break
        print("Ошибка: выберите режим 1 или 2.")

    print("\n" + "-" * 85)
    print(
        "Длина ключа F должна удовлетворять условию: "
        "F=b*N*K, где b - количество бит при переводе в веса "
        "нейронной сети, N - количество входов в нейроны скрытого слоя, "
        "K - количество нейронов скрытого слоя"
    )
    print("-" * 85)

    Bits = read_positive_int("\nВведите b (количество бит): ")
    N = read_positive_int("Введите N (количество входов): ")
    K = read_positive_int("Введите K (количество нейронов): ")

    F = Bits * N * K

    print("\nПараметры нейронной сети:")
    print(f"b = {Bits}")
    print(f"N = {N}")
    print(f"K = {K}")
    print(f"Длина ключа F = {F}")

    print("\n" + "-" * 85)
    print("НАСТРОЙКА ПАРАМЕТРОВ ЭКСПЕРИМЕНТА")
    print("-" * 85)

    qber = read_float(
        "\nВведите QBER [0.05]: ",
        default=0.05,
        min_value=0.0,
        max_value=1.0
    )
    measurements = read_positive_int(
        "Введите количество репликаций [10000]: ",
        default=10000
    )

    if mode == 1:
        default_max_iterations = F
    else:
        default_max_iterations = min(10000, F)

    max_iterations = read_bounded_positive_int(
        f"Введите максимальное количество итераций "
        f"(не более F={F}) [{default_max_iterations}]: ",
        max_value=F,
        default=default_max_iterations
    )

    print("\nВыбранные параметры эксперимента:")
    print(f"QBER = {qber}")
    print(f"Количество репликаций = {measurements}")
    print(f"Максимальное количество итераций = {max_iterations}")

    if mode == 1:
        print("\n" + "=" * 85)
        print("ИССЛЕДОВАНИЕ РАСПРЕДЕЛЕНИЯ ИТЕРАЦИЙ СИНХРОНИЗАЦИИ")
        print("=" * 85)

        print(f"\nb = {Bits}")
        print(f"N = {N}")
        print(f"K = {K}")
        print(f"Длина ключа F = {F}")
        print(f"QBER = {qber}")
        print(f"Количество репликаций = {measurements}")
        print(f"Максимальное количество итераций = {max_iterations}")
        print("\nЗапуск эксперимента...")

        result = measure_sync_iteration_distribution_parallel(
            K=K,
            N=N,
            Bits=Bits,
            measurements=measurements,
            max_iterations=max_iterations,
            Q=qber
        )

        print("\nЭксперимент завершён.")
        print(f"Всего репликаций: {result['measurements']}")
        print(f"Успешных синхронизаций: {result['synchronized_count']}")
        print(f"Неуспешных репликаций: {result['failed_count']}")
        print(f"Файл результатов: {result['output_file']}")

    elif mode == 2:
        print("\n" + "=" * 85)
        print("ИССЛЕДОВАНИЕ ПРОЦЕДУР ПРОВЕРКИ СИНХРОНИЗАЦИИ")
        print("=" * 85)

        print("\nВыберите процедуру:")
        print("1. Без процедуры проверки (none)")
        print("2. Счётчик совпадений выходов (counter)")
        print("3. Отпечатки весовых коэффициентов (hash)")
        print("4. Гибридная процедура (hybrid)")

        while True:
            procedure = input("\nВведите номер процедуры (1-4): ").strip()
            if procedure in ("1", "2", "3", "4"):
                procedure = int(procedure)
                break
            print("Ошибка: выберите процедуру от 1 до 4.")

        sync_modes = {
            1: "none",
            2: "counter",
            3: "hash",
            4: "hybrid"
        }
        sync_check = sync_modes[procedure]

        kStart = 1
        kLimit = 4
        TCounterLimit = 80

        if sync_check == "counter":
            print("\nНастройка счётчика совпадений выходов.")
            TCounterLimit = read_positive_int(
                "Введите TCounterLimit [80]: ",
                default=80
            )
        elif sync_check == "hash":
            print("\nНастройка процедуры отпечатков весовых коэффициентов.")
            kLimit = read_positive_int(
                "Введите kLimit [4]: ",
                default=4
            )
        elif sync_check == "hybrid":
            print("\nНастройка гибридной процедуры.")
            kStart = read_positive_int(
                "Введите kStart [1]: ",
                default=1
            )
            kLimit = read_positive_int(
                "Введите kLimit [4]: ",
                default=4
            )
            TCounterLimit = read_positive_int(
                "Введите TCounterLimit [80]: ",
                default=80
            )

        print("\n" + "-" * 85)
        print("ПАРАМЕТРЫ ЭКСПЕРИМЕНТА")
        print("-" * 85)
        print(f"Режим проверки: {sync_check}")
        print(f"b = {Bits}")
        print(f"N = {N}")
        print(f"K = {K}")
        print(f"Длина ключа F = {F}")
        print(f"QBER = {qber}")
        print(f"Количество репликаций = {measurements}")
        print(f"Максимальное количество итераций = {max_iterations}")

        if sync_check == "counter":
            print(f"TCounterLimit = {TCounterLimit}")
        elif sync_check == "hash":
            print(f"kLimit = {kLimit}")
        elif sync_check == "hybrid":
            print(f"kStart = {kStart}")
            print(f"kLimit = {kLimit}")
            print(f"TCounterLimit = {TCounterLimit}")

        print("\nЗапуск эксперимента...")

        result = measure_single_parameter_set_parallel(
            K=K,
            N=N,
            Bits=Bits,
            kStart=kStart,
            kLimit=kLimit,
            TCounterLimit=TCounterLimit,
            measurements=measurements,
            max_iterations=max_iterations,
            Q=qber,
            sync_check=sync_check
        )

        print("\nЭксперимент завершён.")
        print("\nРезультаты:")
        print(f"FER: {result['fer_average']}")
        print(
            "Среднее число итераций фактической синхронизации: "
            f"{result['fact_sync_iterations_average']}"
        )
        print(
            "Среднее число итераций обнаружения синхронизации: "
            f"{result['sync_iterations_average']}"
        )
        print(
            "Среднее раскрытое количество информации: "
            f"{result['disclosed_information_average']}"
        )

        if sync_check != "none":
            print(f"True Positive: {result['truepositive_average']}")
            print(f"False Positive: {result['falsepositive_average']}")

        if sync_check in ("hash", "hybrid"):
            print(
                "Среднее количество переданных сообщений: "
                f"{result['totalCountMessages_avg']}"
            )
            print(
                "Средняя длина переданных отпечатков: "
                f"{result['totalLenMessages_avg']}"
            )

    print("\n" + "=" * 85)
    print("Работа программы завершена.")
    print("=" * 85)


if __name__ == "__main__":
    main()