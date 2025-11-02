from openai import OpenAI
import time


class OpenAIBatchProcessor:
    def __init__(self, api_key):
        client = OpenAI(api_key=api_key)
        self.client = client

    def process_batch(self, input_file_path, result_file_name, endpoint, completion_window):
        # Upload the input file
        with open(input_file_path, "rb") as file:
            uploaded_file = self.client.files.create(
                file=file,
                purpose="batch"
            )

        # Create the batch job
        batch_job = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint=endpoint,
            completion_window=completion_window
        )

        # Monitor the batch job status
        while batch_job.status not in ["completed", "failed", "cancelled"]:
            time.sleep(3)  # Wait for 3 seconds before checking the status again
            #print(f"Batch job status: {batch_job.status}...trying again in 3 seconds...")
            batch_job = self.client.batches.retrieve(batch_job.id)

        # Download and save the results
        if batch_job.status == "completed":
            result_file_id = batch_job.output_file_id
            #result = self.client.files.retrieve(result_file_id)
            result_object = self.client.files.content(result_file_id)
            result_text = result_object.text

            with open(result_file_name, "w") as file:
                file.write(result_text)


            #print(result_text)  

            return result_text
        else:
            print(f"Batch job failed with status: {batch_job.status}")
            return None
        
