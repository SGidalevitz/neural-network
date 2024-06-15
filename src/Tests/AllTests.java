package Tests;


import org.junit.runner.RunWith;
import org.junit.runners.Suite;


@RunWith(Suite.class)
@Suite.SuiteClasses({
        MatrixTest.class,
        UtilsTest.class,
        NeuralNetworkTest.class
})
public class AllTests {}