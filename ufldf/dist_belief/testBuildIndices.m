testIndices = buildIndices(4, 25, 2, 5, 3, 2);
assert (testIndices(:,1) == [1; 2; 3; 6; 7; 8; 11; 12; 13]);
assert (testIndices(:,2) == [3; 4; 5; 8; 9; 10; 13; 14; 15]);
assert (testIndices(:,3) == [11; 12; 13; 16; 17; 18; 21; 22; 23]);
assert (testIndices(:,4) == [13; 14; 15; 18; 19; 20; 23; 24; 25]);
