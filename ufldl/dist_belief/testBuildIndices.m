function [] = testBuildIndices()

testIndices = buildIndices(5, 2, 3, 2);
assert(all(testIndices(:,1) == [1; 2; 3; 6; 7; 8; 11; 12; 13]));
assert(all(testIndices(:,2) == [3; 4; 5; 8; 9; 10; 13; 14; 15]));
assert(all(testIndices(:,3) == [11; 12; 13; 16; 17; 18; 21; 22; 23]));
assert(all(testIndices(:,4) == [13; 14; 15; 18; 19; 20; 23; 24; 25]));

end
