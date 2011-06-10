function [] = testIndexedWeightSubset()

  full = [
          41  37  31  29
          23  19  17  13
          11   7   5   3
          ];

  stripes = [
             1 2 2
             2 3 4
             ];

  expected = [
              41 37
              19 17
               7  3
              ];

  actual = indexedWeightSubset(full, stripes);

  assert(all(size(expected) == size(actual)));
  assert(all(expected(:) == actual(:)));
end
