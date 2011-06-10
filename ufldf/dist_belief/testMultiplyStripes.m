function [] = testMultiplyStripes()
  data = [
          41  37  31  29
          23  19  17  13
          11   7   5   3
          ];
  stripes = [
             1 2 1
             2 3 3
             ];
  W = [
       .3 .6
       .4 .2
       .5 .1
       ];
  b = [
       1
       2
       -1
       ];
  expected = [
              (.3*41+.6*23) + 1, (.3*37+.6*19) + 1, (.3*31+.6*17) + 1, (.3*29+.6*13) + 1;
              (.4*23+.2*11) + 2, (.4*19+.2*7) + 2, (.4*17+.2*5) + 2, (.4*13+.2*3) + 2;
              (.5*41+.1*11) - 1, (.5*37+.1*7) - 1, (.5*31+.1*5) - 1, (.5*29+.1*3) - 1;
              ];
  actual = multiplyStripes(W, b, stripes, data);

  assert(all(size(expected) == size(actual)));
  assert(all(expected(:) == actual(:)));
end
