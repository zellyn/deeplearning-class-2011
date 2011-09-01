function [] = testBackMultiplyStripes()

  d1 = 41;
  d2 = 37;
  d3 = 31;
  d4 = 29;
  d5 = 23;
  d6 = 19;
  d7 = 17;
  d8 = 13;
  d9 = 11;
  d10 = 7;
  d11 = 5;
  d12 = 3;

  deltas = [
            d1  d2  d3  d4
            d5  d6  d7  d8
            d9 d10 d11 d12
            ];

  stripes = [
             1 4 7
             2 5 8
             3 6 9
             4 7 10
             ];

  wa = randn();
  wb = randn();
  wc = randn();
  wd = randn();
  we = randn();
  wf = randn();
  wg = randn();
  wh = randn();
  wi = randn();
  wj = randn();
  wk = randn();
  wl = randn();

  W = [
       wa wb wc wd
       we wf wg wh
       wi wj wk wl
       ];

  expected = [
              wa*d1,         wa*d2,         wa*d3,           wa*d4;
              wb*d1,         wb*d2,         wb*d3,           wb*d4;
              wc*d1,         wc*d2,         wc*d3,           wc*d4;
              wd*d1 + we*d5, wd*d2 + we*d6, wd*d3 +  we*d7,  wd*d4 + we*d8;
                      wf*d5,         wf*d6,          wf*d7,          wf*d8;
                      wg*d5,         wg*d6,          wg*d7,          wg*d8;
                      wh*d5 + wi*d9, wh*d6 + wi*d10, wh*d7 + wi*d11, wh*d8 + wi*d12;
                              wj*d9,         wj*d10,         wj*d11,         wj*d12;
                              wk*d9,         wk*d10,         wk*d11,         wk*d12;
                              wl*d9,         wl*d10,         wl*d11,         wl*d12;
              ];
  actual = backMultiplyStripes(W, stripes, deltas);

  assert(all(size(expected) == size(actual)));
  assert(all(expected(:) == actual(:)));
end
