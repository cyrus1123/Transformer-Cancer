import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from FCT_trainer import FCT_Trainer
from modeling_big_bird_modified import BigBirdPreTrainedModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ComplexAsyncModelInterface:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model = BigBirdPreTrainedModel()
        self.training_queue = asyncio.Queue()
        self.inference_queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    async def train_model(self, **kwargs):
        async with self.lock:
            trainer = FCT_Trainer(self.model, **kwargs)
            trainer.setup()
            await asyncio.get_event_loop().run_in_executor(self.executor, trainer.train)
            return trainer.get_model()

    async def infer_model(self, input_text):
        async with self.lock:
            return await asyncio.get_event_loop().run_in_executor(self.executor, self.model.predict, input_text)

    async def training_worker(self):
        while True:
            dataset = await self.training_queue.get()
            trained_model = await self.train_model(dataset=dataset)
            self.model = trained_model
            self.training_queue.task_done()
            logging.info(f"Model trained with dataset {dataset}")

    async def inference_worker(self):
        while True:
            input_text = await self.inference_queue.get()
            result = await self.infer_model(input_text)
            self.inference_queue.task_done()
            logging.info(f"Inference result: {result}")

    async def start_workers(self):
        training_task = asyncio.create_task(self.training_worker())
        inference_task = asyncio.create_task(self.inference_worker())
        await asyncio.gather(training_task, inference_task)

    async def enqueue_task(self, task_type: str, data: Any):
        if task_type == 'TRAIN':
            await self.training_queue.put(data)
        elif task_type == 'INFER':
            await self.inference_queue.put(data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def shutdown(self):
        self.executor.shutdown(wait=True)
        logging.info("Executor shutdown completed.")

async def main():
    complex_model_interface = ComplexAsyncModelInterface()
    await complex_model_interface.start_workers()
    await complex_model_interface.enqueue_task('TRAIN', 'your_training_dataset_here')
    await complex_model_interface.enqueue_task('INFER', 'The quick brown fox jumps over the lazy dog')
    await complex_model_interface.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
