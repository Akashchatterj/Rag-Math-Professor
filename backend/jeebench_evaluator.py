"""
JEEBench Benchmark Evaluation Script
=====================================
Evaluates the Math Professor RAG system against JEEBench dataset
(515 challenging problems from IIT JEE-Advanced exam)
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm
import re

from langchain_openai import ChatOpenAI
from datasets import load_dataset


class JEEBenchEvaluator:
    """Evaluate math RAG system on JEEBench dataset"""
    
    def __init__(self, llm, knowledge_base, mcp_search):
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.mcp_search = mcp_search
        self.results = []
        
    def load_jeebench(self, num_samples: int = 50):
        """Load JEEBench dataset from HuggingFace - Mathematics only (all types including MSQ)"""
        try:
            # JEEBench only has 'test' split
            dataset = load_dataset("daman1209arora/jeebench", split="test")
            
            print(f"📊 Total dataset size: {len(dataset)}")
            
            # Check what subjects exist
            subjects = {}
            for item in dataset:
                subj = str(item.get("subject", "Unknown"))
                subjects[subj] = subjects.get(subj, 0) + 1
            
            print(f"📋 Available subjects: {subjects}")
            
            # Filter for Mathematics questions (case-insensitive, handle variations)
            def is_math_question(x):
                subject = str(x.get("subject", "")).lower().strip()
                # Handle different spellings/capitalizations
                return "math" in subject or subject == "mathematics"
            
            math_dataset = dataset.filter(is_math_question)
            
            print(f"✅ Loaded {len(math_dataset)} Mathematics questions from JEEBench")
            print(f"   (Filtered out Physics and Chemistry)")
            
            if len(math_dataset) == 0:
                print("⚠️ No mathematics questions found!")
                print("   Available subjects were:", list(subjects.keys()))
                return self._create_sample_jeebench()
            
            # Show question types
            types = {}
            for item in math_dataset:
                qtype = item.get("type", "Unknown")
                types[qtype] = types.get(qtype, 0) + 1
            print(f"📝 Question types: {types}")
            
            # Limit to num_samples for faster testing
            return math_dataset.select(range(min(num_samples, len(math_dataset))))
            
        except Exception as e:
            print(f"Error loading JEEBench: {e}")
            import traceback
            traceback.print_exc()
            return self._create_sample_jeebench()
    
    def _create_sample_jeebench(self):
        """Create sample JEE-style problems if dataset unavailable"""
        print("Using sample JEE-style problems for evaluation...")
        sample_problems = [
            {
                "question": "If the roots of the equation x² - px + q = 0 are in the ratio 2:3, express p² in terms of q.",
                "answer": "25q/6",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Algebra"
            },
            {
                "question": "Find the value of ∫(0 to π/2) sin²(x) dx",
                "answer": "π/4",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Calculus"
            },
            {
                "question": "What is the maximum value of sin(x) + cos(x)?",
                "answer": "√2",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Trigonometry"
            },
            {
                "question": "Find the coefficient of x² in the expansion of (1 + x)⁵",
                "answer": "10",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Binomial Theorem"
            },
            {
                "question": "If log₂(x) = 3, find the value of x",
                "answer": "8",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Logarithms"
            },
            {
                "question": "The distance between points (3, 4) and (0, 0) is:",
                "answer": "5",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Coordinate Geometry"
            },
            {
                "question": "Find the derivative of e^(2x)",
                "answer": "2e^(2x)",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Calculus"
            },
            {
                "question": "What is the sum of first 10 natural numbers?",
                "answer": "55",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Arithmetic Progression"
            },
            {
                "question": "If vectors a and b are perpendicular, then a·b equals:",
                "answer": "0",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Vectors"
            },
            {
                "question": "The number of ways to arrange 4 distinct objects is:",
                "answer": "24",
                "subject": "Mathematics",
                "type": "Single Correct",
                "topic": "Permutations"
            }
        ]
        return sample_problems
    
    def extract_final_answer(self, solution_text: str) -> str:
        """Extract the final answer from generated solution and clean formatting"""
        # Clean the solution text
        solution_text = solution_text.strip()
        
        # Multiple patterns to catch different formats
        patterns = [
            r"Final Answer:\s*[\\$]*\s*(.+?)(?:\s*[\\$]*)?(?:\n|$)",
            r"Answer:\s*[\\$]*\s*(.+?)(?:\s*[\\$]*)?(?:\n|$)",
            r"Therefore,?\s*(?:the answer is|answer is|we get|result is)\s*[\\$]*\s*(.+?)(?:\s*[\\$]*)?(?:\n|$)",
            r"Thus,?\s*(?:the answer is|answer is|we get)\s*[\\$]*\s*(.+?)(?:\s*[\\$]*)?(?:\n|$)",
            r"Hence,?\s*(?:the answer is|answer is)\s*[\\$]*\s*(.+?)(?:\s*[\\$]*)?(?:\n|$)",
            r"Result:\s*[\\$]*\s*(.+?)(?:\s*[\\$]*)?(?:\n|$)",
            r"Solution:\s*[\\$]*\s*(.+?)(?:\s*[\\$]*)?(?:\n|$)",
            r"####\s*(.+?)(?:\n|$)",  # GSM8K format
            r"\\boxed\{(.+?)\}",  # LaTeX boxed format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution_text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                # Clean up LaTeX and special characters
                answer = answer.replace('$', '').replace('\\', '').strip()
                # Remove markdown bold/italic
                answer = answer.replace('**', '').replace('*', '').replace('__', '').replace('_', '').strip()
                return answer
        
        # Fallback: Try to extract number or option letter from last few lines
        lines = [l.strip() for l in solution_text.split('\n') if l.strip()]
        if lines:
            for line in reversed(lines[-3:]):  # Check last 3 lines
                # Remove markdown formatting from line
                clean_line = line.replace('**', '').replace('*', '').replace('__', '').replace('_', '').strip()
                
                # Look for standalone option letters (A, B, C, D) or numbers
                if re.match(r'^[A-D]$', clean_line, re.IGNORECASE):
                    return clean_line.upper()
                
                # Look for patterns like "(A)" or "A)" or "Option A"
                option_match = re.search(r'\(?\s*([A-D])\s*\)?', clean_line, re.IGNORECASE)
                if option_match:
                    return option_match.group(1).upper()
                
                # Look for standalone numbers or simple expressions
                if re.match(r'^[\d\.\-\+\*/\^\(\)\s]+$', clean_line) or len(clean_line) < 30:
                    return clean_line
        
        # Last resort: return last line (cleaned)
        if lines:
            return lines[-1].replace('**', '').replace('*', '').strip()
        
        return ""
    
    def evaluate_answer(self, generated: str, ground_truth: str) -> Dict:
        """
        Evaluate generated answer against ground truth
        Handles MCQ, MSQ (multiple correct), Integer, and Numeric types
        Returns: Dict with correctness metrics
        """
        # Normalize answers - remove spaces and special chars
        gen_norm = generated.lower().strip().replace(' ', '').replace(',', '')
        gt_norm = ground_truth.lower().strip().replace(' ', '').replace(',', '')
        
        # Remove common formatting
        for char in ['$', '\\', '(', ')', '{', '}', '[', ']', '.']:
            gen_norm = gen_norm.replace(char, '')
            gt_norm = gt_norm.replace(char, '')
        
        # Exact match (after normalization)
        exact_match = gen_norm == gt_norm
        
        # For MCQ: Check if ground truth letters are in generated
        import string
        gt_letters = sorted([c for c in gt_norm if c in string.ascii_lowercase])
        gen_letters = sorted([c for c in gen_norm if c in string.ascii_lowercase])
        
        # MSQ handling: Check if all ground truth options are present
        msq_match = False
        if gt_letters:
            # All ground truth letters must be in generated, and no extra letters
            msq_match = (set(gt_letters) == set(gen_letters))
            
            # Partial MSQ: At least some correct options identified
            partial_msq = len(set(gt_letters) & set(gen_letters)) > 0
        else:
            partial_msq = False
        
        # Check if ground truth is substring of generated (for longer answers)
        contains_match = gt_norm in gen_norm
        
        # Extract and compare numbers (for Integer/Numeric types)
        numbers_gen = re.findall(r'-?\d+\.?\d*', generated)
        numbers_gt = re.findall(r'-?\d+\.?\d*', ground_truth)
        
        numerical_match = False
        if numbers_gen and numbers_gt:
            # Check if all ground truth numbers appear
            numerical_match = all(num in numbers_gen for num in numbers_gt)
            
            # Also check for numerical equivalence (within tolerance)
            if not numerical_match and len(numbers_gt) == 1 and len(numbers_gen) >= 1:
                try:
                    gt_num = float(numbers_gt[0])
                    # Check if any generated number is close to ground truth
                    numerical_match = any(abs(float(gn) - gt_num) < 0.01 for gn in numbers_gen)
                except:
                    pass
        
        # Combined partial match
        partial_match = contains_match or msq_match or partial_msq or numerical_match
        
        # Semantic match (use LLM only if not already matched)
        semantic_match = exact_match or msq_match or partial_match
        if not semantic_match:
            semantic_match = self._check_semantic_match(generated, ground_truth)
        
        # Scoring
        if exact_match or msq_match:
            score = 1.0
        elif partial_match or partial_msq:
            score = 0.7
        elif semantic_match:
            score = 0.5
        else:
            score = 0.0
        
        return {
            "exact_match": exact_match or msq_match,
            "partial_match": partial_match,
            "semantic_match": semantic_match,
            "score": score
        }
    
    def _check_semantic_match(self, generated: str, ground_truth: str) -> bool:
        """Use LLM to check if answers are semantically equivalent"""
        try:
            prompt = f"""
            Compare these two mathematical answers and determine if they are semantically equivalent.
            
            Answer 1: {generated}
            Answer 2: {ground_truth}
            
            Consider:
            - Mathematical equivalence (e.g., 1/2 = 0.5)
            - Different representations of the same value
            - Simplified vs. expanded forms
            
            Respond with ONLY "YES" if equivalent or "NO" if different.
            """
            
            response = self.llm.invoke(prompt)
            return "YES" in response.content.upper()
        except:
            return False
    
    def solve_problem(self, problem: Dict) -> Dict:
        """Solve a single JEEBench problem"""
        question = problem.get("question", "")
        
        # Step 1: Check knowledge base
        kb_results = self.knowledge_base.search(question, k=3)
        
        if kb_results:
            context_source = "knowledge_base"
            context = "\n\n".join([r['content'] for r in kb_results])
        else:
            context_source = "web_search"
            web_results = self.mcp_search.search(question)
            if web_results.get('results'):
                context = "\n\n".join([str(r) for r in web_results['results']])
            else:
                context = ""
        
        # Step 2: Generate solution with better prompting for JEE questions
        solution_prompt = f"""
You are solving a JEE Advanced Mathematics problem. Think step-by-step and be precise.

**Problem:** {question}

**Context:**
{context}

**Instructions:**
1. Read the problem carefully and identify what is being asked
2. Show your mathematical reasoning step-by-step
3. For MCQ: Evaluate each option if helpful
4. For MCQ(multiple): Identify ALL correct options
5. Calculate precisely - double-check your arithmetic
6. State your final answer CLEARLY

**CRITICAL: Your final answer MUST be in this exact format:**
- For MCQ with single answer: State just the letter (A, B, C, or D)
- For MCQ(multiple): State all correct letters separated by commas (e.g., A,C or B,D)
- For Integer/Numeric: State just the number (e.g., 25 or 0.5)

**Format your response as:**

**Step-by-Step Solution:**
[Show your detailed work here]

**Final Answer:** [Just the letter(s) or number - nothing else]

Remember: JEE problems require careful calculation. Double-check your work!
"""
        
        try:
            response = self.llm.invoke(solution_prompt)
            solution = response.content
            
            # Extract final answer with better handling
            predicted_answer = self.extract_final_answer(solution)
            
            return {
                "question": question,
                "context_source": context_source,
                "solution": solution,
                "predicted_answer": predicted_answer,
                "success": True
            }
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "success": False
            }
    
    def run_benchmark(self, num_samples: int = 50, save_results: bool = True):
        """Run full benchmark evaluation with detailed debugging and progress tracking"""
        print(f"🚀 Starting JEEBench Evaluation ({num_samples} samples - Mathematics only)")
        print("=" * 70)
        
        # Load dataset
        dataset = self.load_jeebench(num_samples)
        
        if len(dataset) == 0:
            print("❌ No mathematics questions found in dataset!")
            return {"error": "No data"}
        
        print(f"✅ Loaded {len(dataset)} Mathematics questions")
        
        # Show breakdown by type
        type_counts = {}
        for item in dataset:
            qtype = item.get('type', 'Unknown')
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        print("\n📊 Question Type Breakdown:")
        for qtype, count in sorted(type_counts.items()):
            print(f"   - {qtype}: {count} questions")
        
        print("=" * 70)
        
        # Track metrics
        total_score = 0
        correct_exact = 0
        correct_partial = 0
        correct_semantic = 0
        
        # For debugging: track first few comparisons
        debug_count = min(3, len(dataset))
        
        # Progress tracking
        save_interval = 50  # Save every 50 questions
        start_time = datetime.now()
        
        # Evaluate each problem
        for i, problem in enumerate(tqdm(dataset, desc="Evaluating")):
            # Solve problem
            result = self.solve_problem(problem)
            
            if result['success']:
                # CRITICAL: Use 'gold' field from JEEBench, not 'answer'
                ground_truth = problem.get('gold', problem.get('answer', ''))
                predicted = result['predicted_answer']
                
                # Skip if no ground truth
                if not ground_truth or str(ground_truth).strip() == '':
                    print(f"\n⚠️ Skipping question {i+1}: No ground truth answer")
                    continue
                
                # Debug first few
                if i < debug_count:
                    print(f"\n--- Debug Sample {i+1} ---")
                    print(f"Question: {problem.get('question', '')[:100]}...")
                    print(f"Type: {problem.get('type', 'Unknown')}")
                    print(f"Ground Truth: '{ground_truth}'")
                    print(f"Predicted: '{predicted}'")
                
                # Evaluate answer
                evaluation = self.evaluate_answer(predicted, ground_truth)
                
                if i < debug_count:
                    print(f"Exact Match: {evaluation['exact_match']}")
                    print(f"Partial Match: {evaluation['partial_match']}")
                    print(f"Semantic Match: {evaluation['semantic_match']}")
                    print(f"Score: {evaluation['score']}")
                    print("---\n")
                
                # Update metrics
                total_score += evaluation['score']
                correct_exact += int(evaluation['exact_match'])
                correct_partial += int(evaluation['partial_match'])
                correct_semantic += int(evaluation['semantic_match'])
                
                # Store result
                self.results.append({
                    **result,
                    "ground_truth": ground_truth,
                    "evaluation": evaluation,
                    "subject": problem.get('subject', ''),
                    "type": problem.get('type', ''),
                    "topic": problem.get('topic', ''),
                    "problem_id": i
                })
                
                # Periodic progress update
                if (i + 1) % 25 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (len(dataset) - i - 1)
                    
                    current_exact = correct_exact / (i + 1)
                    current_partial = correct_partial / (i + 1)
                    
                    print(f"\n📊 Progress Update [{i+1}/{len(dataset)}]:")
                    print(f"   Exact Match: {current_exact:.1%}")
                    print(f"   Partial Match: {current_partial:.1%}")
                    print(f"   Estimated time remaining: {remaining/60:.1f} minutes\n")
                
                # Periodic save (every 50 questions)
                if (i + 1) % save_interval == 0 and save_results:
                    self._save_intermediate_results(i + 1)
        
        # Calculate final metrics
        num_evaluated = len(self.results)
        
        if num_evaluated == 0:
            print("❌ No problems were successfully evaluated!")
            return {"error": "No results"}
        
        metrics = {
            "total_problems": num_evaluated,
            "exact_match_accuracy": correct_exact / num_evaluated,
            "partial_match_accuracy": correct_partial / num_evaluated,
            "semantic_match_accuracy": correct_semantic / num_evaluated,
            "average_score": total_score / num_evaluated,
            "timestamp": datetime.now().isoformat(),
            "question_types": type_counts,
            "evaluation_time_seconds": (datetime.now() - start_time).total_seconds()
        }
        
        # Print results
        self._print_results(metrics)
        
        # Save results
        if save_results:
            self._save_results(metrics)
        
        return metrics
    
    def _save_intermediate_results(self, num_completed: int):
        """Save intermediate results during long evaluations"""
        results_dir = Path("./benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = results_dir / f"intermediate_{num_completed}q_{timestamp}.json"
        
        with open(intermediate_file, 'w') as f:
            json.dump({
                "num_completed": num_completed,
                "results": self.results
            }, f, indent=2)
        
        print(f"💾 Intermediate results saved: {num_completed} questions completed")
    
    def _print_results(self, metrics: Dict):
        """Print benchmark results with detailed breakdown"""
        print("\n" + "=" * 70)
        print("📊 BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Total Problems Evaluated: {metrics['total_problems']}")
        print(f"Evaluation Time: {metrics.get('evaluation_time_seconds', 0)/60:.1f} minutes")
        print(f"\n📈 Overall Accuracy:")
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
        print(f"Partial Match Accuracy: {metrics['partial_match_accuracy']:.2%}")
        print(f"Semantic Match Accuracy: {metrics['semantic_match_accuracy']:.2%}")
        print(f"Average Score: {metrics['average_score']:.2f}")
        
        # Breakdown by question type
        if 'question_types' in metrics:
            print(f"\n📋 Performance by Question Type:")
            
            type_stats = {}
            for result in self.results:
                qtype = result.get('type', 'Unknown')
                if qtype not in type_stats:
                    type_stats[qtype] = {
                        'total': 0,
                        'exact': 0,
                        'partial': 0,
                        'score_sum': 0
                    }
                
                type_stats[qtype]['total'] += 1
                type_stats[qtype]['exact'] += int(result['evaluation']['exact_match'])
                type_stats[qtype]['partial'] += int(result['evaluation']['partial_match'])
                type_stats[qtype]['score_sum'] += result['evaluation']['score']
            
            for qtype in sorted(type_stats.keys()):
                stats = type_stats[qtype]
                exact_pct = stats['exact'] / stats['total'] if stats['total'] > 0 else 0
                partial_pct = stats['partial'] / stats['total'] if stats['total'] > 0 else 0
                avg_score = stats['score_sum'] / stats['total'] if stats['total'] > 0 else 0
                
                print(f"\n   {qtype} ({stats['total']} questions):")
                print(f"      Exact Match: {exact_pct:.1%}")
                print(f"      Partial Match: {partial_pct:.1%}")
                print(f"      Avg Score: {avg_score:.2f}")
        
        print("=" * 70)
    
    def _save_results(self, metrics: Dict):
        """Save benchmark results to file"""
        results_dir = Path("./benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = results_dir / f"jeebench_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "metrics": metrics,
                "detailed_results": self.results
            }, f, indent=2)
        
        # Save summary
        summary_file = results_dir / f"jeebench_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("JEEBench Benchmark Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Date: {metrics['timestamp']}\n")
            f.write(f"Total Problems: {metrics['total_problems']}\n")
            f.write(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}\n")
            f.write(f"Partial Match Accuracy: {metrics['partial_match_accuracy']:.2%}\n")
            f.write(f"Semantic Match Accuracy: {metrics['semantic_match_accuracy']:.2%}\n")
            f.write(f"Average Score: {metrics['average_score']:.2f}\n")
        
        print(f"\n✅ Results saved to:")
        print(f"   - {results_file}")
        print(f"   - {summary_file}")


def main():
    """Main execution function"""
    import argparse
    
    # Add command line arguments
    parser = argparse.ArgumentParser(description='JEEBench Mathematics Evaluator')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of samples to evaluate (default: all 236 math questions)')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with only 10 samples')
    args = parser.parse_args()
    
    # Import from main app
    from math_professor_rag import Config, MathKnowledgeBase, MCPWebSearchTool
    from langchain_openai import ChatOpenAI
    
    # Initialize configuration
    config = Config()
    
    # Initialize LLM
    llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    
    # Initialize knowledge base and search
    print("Initializing knowledge base...")
    knowledge_base = MathKnowledgeBase()
    knowledge_base.load_datasets()
    
    mcp_search = MCPWebSearchTool(config.TAVILY_API_KEY)
    
    # Create evaluator
    evaluator = JEEBenchEvaluator(llm, knowledge_base, mcp_search)
    
    # Determine number of samples
    if args.quick_test:
        num_samples = 10
        print("\n🚀 Quick Test Mode: Evaluating 10 samples")
    elif args.num_samples:
        num_samples = args.num_samples
        print(f"\n🚀 Custom Mode: Evaluating {num_samples} samples")
    else:
        num_samples = 236  # ALL mathematics questions
        print("\n🚀 Full Evaluation Mode: Evaluating ALL 236 mathematics questions")
        print("⏱️ This will take approximately 30-40 minutes...")
        
        # Confirm for full run
        response = input("\n⚠️ Continue with full evaluation? (y/n): ")
        if response.lower() != 'y':
            print("❌ Evaluation cancelled")
            return None
    
    # Run benchmark
    print(f"\n▶️ Starting evaluation of {num_samples} questions...")
    metrics = evaluator.run_benchmark(num_samples=num_samples)
    
    return metrics


if __name__ == "__main__":
    main()