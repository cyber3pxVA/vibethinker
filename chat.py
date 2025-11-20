from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os


class VibeThinker:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Loading model from {model_path} on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            torch_dtype="float32",
            device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print("Model loaded successfully on CPU!\n")

    def infer_text(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generation_config = dict(
            max_new_tokens=40960,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=None
        )
        generated_ids = self.model.generate(
            **model_inputs,
            generation_config=GenerationConfig(**generation_config)
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


if __name__ == '__main__':
    # Initialize the model from Hugging Face
    model = VibeThinker('WeiboAI/VibeThinker-1.5B')
    
    print("=" * 60)
    print("VibeThinker-1.5B Interactive Chat")
    print("=" * 60)
    print("Best for: Math problems and coding challenges")
    print("Type 'exit' or 'quit' to end the chat\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
                
            if not user_input:
                continue
            
            print("\nVibeThinker: ", end="", flush=True)
            response = model.infer_text(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
