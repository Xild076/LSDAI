import openai
import signal
import datetime
import time
from colorama import Fore
from utility import get_api_key
openai.api_key = get_api_key('api_key.txt')

def fine_tune(training_file, model, epochs, log=False):
    if log:
        print('Creating file on OpenAI... \r')
    
    training_file_id = openai.files.create(
        file=open(training_file, 'rb'),
        purpose='fine-tune'
    )
    
    if log:
        print(Fore.GREEN + 'SUCCEESS!' + Fore.RESET)
        print(f'File ID: {training_file_id.id}')
        print('Creating Fine-Tuning Job... \r')
    
    response = openai.fine_tuning.jobs.create(
        training_file=training_file_id.id,
        model=model,
        hyperparameters={
            'n_epochs': epochs
        }
    )
    job_id = response.id
    status = response.status
    
    if log:
        print(Fore.GREEN + 'SUCCESS!' + Fore.RESET)
        print(f'Job ID: {job_id}')
        print(f'Job Status: {status}')
    
    def signal_handler(sig, frame):
        status = openai.fine_tuning.jobs.retrieve(job_id).status
        print(f'Stream interrupted. Job is still {status}.')
        return
    
    if log:
        signal.signal(signal.SIGINT, signal_handler)

        events = openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
        try:
            for event in events:
                print(
                    f'{datetime.datetime.fromtimestamp(event.created_at)} {event.message}'
                )
        except Exception:
            print(Fore.RED + 'Stream interrupted (client disconnected).' + Fore.RESET)
    
    status = openai.fine_tuning.jobs.retrieve(job_id).status
    if status not in ["succeeded", "failed"]:
        
        if log:
            print(f"Job not in terminal status: {status}. Waiting.")
        
        while status not in ["succeeded", "failed"]:
            time.sleep(2)
            status = openai.fine_tuning.jobs.retrieve(job_id).status
    else:
        if log:
            print(f"Finetune job {job_id} finished with status: {status}")
    
    if log:
        print("Checking other finetune jobs in the subscription.")
    
    result = openai.fine_tuning.jobs.list()
    
    if log:
        print(f"Found {len(result.data)} finetune jobs.")
    
    result = openai.fine_tuning.jobs.list()
    mm = result.data[0].fine_tuned_model

    if log:
        print(f'Model ({mm}) created successfully!')

    return mm

def generate_response(model, prompt):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()