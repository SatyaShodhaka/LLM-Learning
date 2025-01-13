from abc import ABC, abstractmethod
import random

class Batcher(ABC):
   @abstractmethod
   def batch(self, data, batch_size): pass
   def get_description(self): pass


class GradientBatcher(Batcher):
   def batch(self, data, batch_size):
      num_batched = len(data) // batch_size
      return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
   
   
   def get_description(self):
      return "Gradient batcher"
   

class BucketBatcher(Batcher):
   def __init__(self, splits, repetitions, randomize=False, seed=0):
      '''
      Params:
      splits: list of numbers summing up to 1.0 representing % of 
              datapoints corresponding to difficulty level
      repetitions: list of numbers > 0 representing number of times
                   datapoints in a split should be repeated; length
                   must be equal to number of splits
      randomize: randomize datapoints within buckets
      seed: seed for random, if randomize=True and if desired
      '''

      self.bucket_splits = splits
      self.bucket_repetitions = repetitions
      self.randomize = randomize
      self.seed = seed
      random.seed(seed)
      

   def batch(self, data, batch_size):
      batches = []

      split_sizes = [int(len(data) * split_percentage + 0.5) for split_percentage in self.bucket_splits]    
      lower_bound = 0
      for idx, split_size in enumerate(split_sizes):
         split_data = data[lower_bound:min(lower_bound + split_size, len(data))]
         for _ in range(self.bucket_repetitions[idx]):
            if self.randomize:
               random.shuffle(split_data)
            batches.extend([split_data[i:i+batch_size] for i in range(0, len(split_data), batch_size)])
         lower_bound += split_size
      
      return batches

   
   def get_description(self):
      settings = " - ".join(
         [f"{int(split*100)}% ({repetition}x)" for split, repetition in zip(self.bucket_splits, self.bucket_repetitions)]
      )
      return "Bucket Batcher in " + settings

