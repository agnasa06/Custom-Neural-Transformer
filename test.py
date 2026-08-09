import time

import torch
import torch_directml


device = torch_directml.device()


def synchronize(tensor: torch.Tensor) -> None:
	tensor.reshape(-1)[0].cpu().item()


def measure(
	a: torch.Tensor,
	b: torch.Tensor,
	repetitions: int,
) -> float:
	# More warmup so operator compilation and caching settle.
	for _ in range(50):
		c = a @ b

	synchronize(c)

	start = time.perf_counter()

	for _ in range(repetitions):
		c = a @ b

	synchronize(c)

	return (time.perf_counter() - start) / repetitions


def benchmark_shape(
	rows: int,
	inner: int,
	columns: int,
	repetitions: int = 50,
	rounds: int = 6,
) -> None:
	print(f"\n[{rows}, {inner}] @ [{inner}, {columns}]")

	tensors = {}

	for dtype in (torch.float32, torch.float16):
		tensors[dtype] = (
			torch.randn(
				rows,
				inner,
				dtype=dtype,
				device=device,
			),
			torch.randn(
				inner,
				columns,
				dtype=dtype,
				device=device,
			),
		)

	results = {
		torch.float32: [],
		torch.float16: [],
	}

	for round_number in range(rounds):
		# Alternate order to prevent the first dtype from consistently
		# paying some one-time or cache-related cost.
		if round_number % 2 == 0:
			order = (torch.float32, torch.float16)
		else:
			order = (torch.float16, torch.float32)

		for dtype in order:
			a, b = tensors[dtype]

			average_time = measure(
				a,
				b,
				repetitions,
			)

			results[dtype].append(average_time)

			print(
				f"Round {round_number + 1}, {dtype}: "
				f"{average_time * 1000:.4f} ms"
			)

	# Ignore the first result for each dtype in case some setup remains.
	fp32_times = results[torch.float32][1:]
	fp16_times = results[torch.float16][1:]

	fp32_average = sum(fp32_times) / len(fp32_times)
	fp16_average = sum(fp16_times) / len(fp16_times)

	print(f"Stable FP32 average: {fp32_average * 1000:.4f} ms")
	print(f"Stable FP16 average: {fp16_average * 1000:.4f} ms")
	print(f"Stable FP16 speedup: {fp32_average / fp16_average:.3f}x")


benchmark_shape(
	rows=128 * 128,
	inner=384,
	columns=1536,
	repetitions=30,
)

benchmark_shape(
	rows=128 * 128,
	inner=1536 * 16,
	columns=384,
	repetitions=30,
)