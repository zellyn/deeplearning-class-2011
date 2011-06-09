function [] = testAll()

  disp('Checking buildIndices...');
  testBuildIndices();
  disp('Done.');

  disp('Checking unpackTheta...');
  testUnpackTheta();
  disp('Done.');

  disp('Checking multiplyStripes...');
  testMultiplyStripes();
  disp('Done.');

  disp('Checking backMultiplyStripes...');
  testBackMultiplyStripes();
  disp('Done.');

  disp('Checking indexedWeightSubset...');
  testIndexedWeightSubset();
  disp('Done.');

  disp('Checking cost func mlpCost...');
  checkMlpCost();
  disp('Done.');
end
