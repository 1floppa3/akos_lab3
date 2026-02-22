import threading
import queue
import time
import random
import sys
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import signal


@dataclass
class WorkItem:
    item_id: int
    content: str
    origin: str
    created_at: float = field(default_factory=time.time)


@dataclass
class ProcessingOutput:
    item_id: int
    word_counts: Dict[str, int]
    total_terms: int
    unique_terms: int
    worker_id: int
    duration: float


class ContentWorker(threading.Thread):
    def __init__(self, worker_id: int, pending_queue: queue.Queue,
                 global_counter: Dict[str, int], counter_lock: threading.Lock,
                 termination_token: object):
        super().__init__()
        self.worker_id = worker_id
        self.pending_queue = pending_queue
        self.global_counter = global_counter
        self.counter_lock = counter_lock
        self.termination_token = termination_token
        self.processed_count = 0
        self.total_work_time = 0
        self.terms_handled = 0
        self.daemon = True

    def sanitize_text(self, raw_text: str) -> List[str]:
        clean = raw_text.lower()
        clean = re.sub(r'[^\w\s]', ' ', clean)
        clean = re.sub(r'\d+', ' ', clean)
        tokens = [token for token in clean.split() if len(token) > 1]
        return tokens

    def analyze_content(self, raw_text: str) -> Tuple[Dict[str, int], int, int]:
        all_tokens = self.sanitize_text(raw_text)
        total_tokens = len(all_tokens)

        frequency_map = {}
        for token in all_tokens:
            frequency_map[token] = frequency_map.get(token, 0) + 1

        unique_tokens = len(frequency_map)

        return frequency_map, total_tokens, unique_tokens

    def update_global_stats(self, local_counts: Dict[str, int]):
        with self.counter_lock:
            for term, occurrences in local_counts.items():
                self.global_counter[term] = self.global_counter.get(
                    term, 0) + occurrences

    def run(self):
        while True:
            try:
                assignment = self.pending_queue.get(timeout=1)

                if assignment is self.termination_token:
                    self.pending_queue.task_done()
                    break

                start_moment = time.time()

                word_freq, total_terms, unique_terms = self.analyze_content(
                    assignment.content)

                self.update_global_stats(word_freq)

                elapsed = time.time() - start_moment
                self.processed_count += 1
                self.total_work_time += elapsed
                self.terms_handled += total_terms

                self.pending_queue.task_done()

                time.sleep(random.uniform(0.01, 0.03))

            except queue.Empty:
                continue
            except Exception as e:
                self.pending_queue.task_done()


class AssignmentGenerator(threading.Thread):
    def __init__(self, target_queue: queue.Queue, total_assignments: int,
                 termination_marker: object, source_text: str):
        super().__init__()
        self.target_queue = target_queue
        self.total_assignments = total_assignments
        self.termination_marker = termination_marker
        self.source_text = source_text
        self.generated_count = 0
        self.daemon = True

    def partition_text(self, text_body: str, parts: int) -> List[str]:
        words = text_body.split()
        chunk_dimension = len(words) // parts
        if chunk_dimension == 0:
            chunk_dimension = 1

        fragments = []
        for idx in range(parts):
            start_idx = idx * chunk_dimension
            if idx == parts - 1:
                fragment = ' '.join(words[start_idx:])
            else:
                fragment = ' '.join(
                    words[start_idx:start_idx + chunk_dimension])
            if fragment.strip():
                fragments.append(fragment)

        return fragments

    def run(self):
        text_fragments = self.partition_text(
            self.source_text, self.total_assignments)

        for idx in range(min(self.total_assignments, len(text_fragments))):
            new_task = WorkItem(
                item_id=idx + 1,
                content=text_fragments[idx],
                origin=f"segment_{idx}"
            )

            self.target_queue.put(new_task)
            self.generated_count += 1

            time.sleep(random.uniform(0.1, 0.3))


class ProcessingOrchestrator:
    def __init__(self, worker_pool_size: int = 3, total_jobs: int = 20):
        self.worker_pool_size = worker_pool_size
        self.total_jobs = total_jobs

        self.job_buffer = queue.Queue(maxsize=10)

        self.master_counter = defaultdict(int)
        self.counter_protector = threading.Lock()

        self.shutdown_indicator = object()

        self.generator = None
        self.worker_threads = []

        self.analysis_subject = """
При разработке приложений часто возникает одна и та же проблема: у каждого разработчика своё окружение. Разные версии библиотек, интерпретаторов, системных зависимостей. В результате код, который стабильно работает на одном компьютере, может не запуститься на другом.
Docker решает эту проблему — он создаёт единый и воспроизводимый способ запускать приложения.
Основная идея: Docker — это платформа контейнеризации. Она позволяет упаковать приложение вместе со всеми зависимостями в изолированную среду — контейнер. 
Такой контейнер можно запустить где угодно: на локальной машине, сервере или в облаке, и результат всегда будет одинаковым.
До появления Docker основным способом изоляции приложений были виртуальные машины. 
Каждая виртуальная машина имитировала полноценный компьютер: со своей ОС, драйверами и файловой системой. Для управления ими использовался гипервизор — программный слой, который распределяет ресурсы хоста между несколькими виртуальными машинами.
Подход надёжный, но тяжёлый. Каждая ВМ занимала сотни мегабайт или гигабайты памяти, запускалась медленно и требовала отдельного обслуживания.
Docker работает иначе. Он использует подход OS-level virtualization — контейнеризацию на уровне операционной системы. Вместо того чтобы поднимать отдельную ОС под каждое приложение, Docker создаёт изолированные контейнеры, которые делят ядро хостовой системы, но имеют собственное пространство процессов, файлов и сетей.
Если упрощённо:
виртуальная машина изолирует железо и поднимает целую операционную систему;
контейнер изолирует процессы в рамках одной ОС.
5 причин, зачем нужен Docker
1. Стабильное окружение. Docker гарантирует, что приложение будет работать одинаково везде — на ноутбуке разработчика, тестовом сервере или в продакшене. Всё, что нужно для запуска, уже собрано в контейнер, поэтому код не зависит от различий в системах и настройках.
2. Лёгкость и скорость. Контейнеры используют общее ядро операционной системы и не создают отдельную копию ОС. За счёт этого они занимают меньше места и запускаются за секунды. Один сервер может без труда обслуживать десятки контейнеров.
3. Масштабирование. При росте нагрузки можно просто запустить дополнительные контейнеры — система быстро увеличит мощность и перераспределит ресурсы без изменения кода.
4. Изоляция процессов. Каждый контейнер работает независимо от других. Если в одном произойдёт сбой или утечка памяти, это не затронет другие контейнеры и систему в целом.
5. Удобная интеграция в CI/CD. Контейнеры уже стали стандартом в современных пайплайнах. Приложение можно собрать, протестировать и запустить в одинаковой среде — от локальной машины до продакшена. Благодаря этому меньше неожиданных багов, а релизы проходят быстрее и спокойнее.
"""

    def register_interrupt_handlers(self):
        def interrupt_handler(sig, frame):
            print("\nInterrupt signal received. Shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, interrupt_handler)

    def launch(self):
        print(f"Worker threads: {self.worker_pool_size}")
        print(f"Total jobs: {self.total_jobs}")
        print("Analysis: word frequency counting")

        for idx in range(self.worker_pool_size):
            worker = ContentWorker(
                worker_id=idx + 1,
                pending_queue=self.job_buffer,
                global_counter=self.master_counter,
                counter_lock=self.counter_protector,
                termination_token=self.shutdown_indicator
            )
            self.worker_threads.append(worker)
            worker.start()

        self.generator = AssignmentGenerator(
            target_queue=self.job_buffer,
            total_assignments=self.total_jobs,
            termination_marker=self.shutdown_indicator,
            source_text=self.analysis_subject
        )
        self.generator.start()

    def await_completion(self):
        self.generator.join()

        for _ in self.worker_threads:
            self.job_buffer.put(self.shutdown_indicator)

        for worker in self.worker_threads:
            worker.join()

        self.job_buffer.join()

    def display_findings(self):
        print("processing results")

        print("\nworker statistics:")
        total_assignments = 0
        cumulative_time = 0
        cumulative_terms = 0

        for worker in self.worker_threads:
            total_assignments += worker.processed_count
            cumulative_time += worker.total_work_time
            cumulative_terms += worker.terms_handled
            print(f"  Worker {worker.worker_id}:")
            print(f"     Assignments: {worker.processed_count}")
            print(f"     Terms: {worker.terms_handled}")
            print(f"     Time: {worker.total_work_time:.3f}s")

        print(f"  total:")
        print(f"    • Assignments: {total_assignments}")
        print(f"    • Terms: {cumulative_terms}")
        print(f"    • Time: {cumulative_time:.3f}s")
        if cumulative_time > 0:
            print(
                f"    • Throughput: {cumulative_terms/cumulative_time:.0f} terms/s")

        print("\nfrequency of words:")
        unique_terms_total = len(self.master_counter)
        total_occurrences = sum(self.master_counter.values())

        print(f"   Unique terms: {unique_terms_total}")
        print(f"   Total occurrences: {total_occurrences}")

        if total_occurrences > 0:
            sorted_terms = sorted(
                self.master_counter.items(), key=lambda x: x[1], reverse=True)

            print("\n  Top 30 most frequent terms:")
            for idx, (term, freq) in enumerate(sorted_terms[:30], 1):
                proportion = (freq / total_occurrences * 100)
                print(
                    f"  {idx:2}. {term:25} → {freq:3} times ({proportion:.1f}%)")

    def shutdown(self):
        pass


def entry_point():
    worker_count = 3
    job_count = 20

    if len(sys.argv) > 1:
        try:
            worker_count = int(sys.argv[1])
        except ValueError:
            pass

    if len(sys.argv) > 2:
        try:
            job_count = int(sys.argv[2])
        except ValueError:
            pass

    system = ProcessingOrchestrator(
        worker_pool_size=worker_count,
        total_jobs=job_count
    )

    system.register_interrupt_handlers()

    try:
        system.launch()
        system.await_completion()
        system.display_findings()

        print("\nProcessing completed successfully")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(entry_point())
