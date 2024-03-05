from evaluations.evaluate import evaluate_response

if __name__ == '__main__':
    case = {
        'input': 'What is the population of Boston?  Give me the exact number and not an approximation.',
        'output': 'The city of Boston has a population of approximately 654,000 people.',
        # 'output': 'The city of Boston has a population of approximately 665,000 people.',
        'context': 'The population of Boston is 654,776 people.',
    }
    evaluate_response(case['input'], case['output'], case['context'])

