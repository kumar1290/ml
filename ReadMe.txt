How to Install CloudSim in Ubuntu (in 3 easy steps)
The contents of this DIY has been tested on Ubuntu 12.04 LTS x64 with CloudSim 3.0.3 beta and JDK8u11 x64. It should work similarly on other distros of Ubuntu and for different version of JDK as long as you use Java version 1.5 or newer. The text shown in RED in the DIY represents the values that will differ depending upon your JDK version and locations.

CloudSim is a framework for modelling and simulating cloud computing infrastructure and services. In this DIY we will see how to install CloudSim in Windows easily and efficiently. A more detailed description, APIs and research papers related to CloudSim can be found here. Let us begin.


Step 1: Setting up the Prerequisites

1. First of all we need to download the CloudSim and latest version of the Java Development Toolkit (JDK). CloudSim can be found here.

2. CloudSim requires a working Java installation. So, open up a terminal and run the following

1	sudo add-apt-repository ppa:webupd8team/java
2	sudo apt-get update && sudo apt-get install oracle-java8-installer

It will take some time to download and install so sit back and wait. Once it’s done then we have to add the JAVA_HOME to the Ubuntu environment. Run the following in a terminal to open up the /etc/environment file.

1	sudo gedit /etc/environment

Now, append the following at the end of the file and save it:

JAVA_HOME="/usr/lib/jvm/java-8-oracle"

3. Now its time to install the CloudSim. Unpack the downloaded 'CloudSim-3.0.3.tar.gz' or 'CloudSim-3.0.3.zip' (let the name of the unpacked folder be 'cloudsim-3.0.3'). As you can see there is no makefile or install file inside the folder so it doesn't need to be compiled. Later if you want to remove the CloudSim, just remove the whole 'cloudsim-3.0.3' directory.


Step 2: Setting up the Environment

Now comes the critical part, the most important part of the CloudSim setup is the setting up the paths to different classes and jar files correctly or you won't be able to run your programs efficiently.

We need to set the 'CLASSPATH' variable which will contain the location of the class files and will be used by the CloudSim while executing an application. So we have to set two consecutive locations first one is the location of gridsim.jar file provided in the CloudSim and is used exclusively by the CloudSim applications and second one is the location where we have stored our programs.

We will set the CLASSPATH in the .bashrc file of the current user so open a terminal and run the following

1	sudo gedit /home/dhyan/.bashrc

Provide the password and add the following lines at the end of the opened file and save it.

CLASSPATH=".:/home/dhyan/Desktop/cloudsim-3.0.3/jars/*:
/home/dhyan/Desktop/cloudsim-3.0.3/examples"
export CLASSPATH

Now we need to reload the .bashrc file so close the all opened terminals (if any) and run the following

1	source ~/.bashrc


Step 3: Testing the Setup (Compiling and Executing a CloudSim Application)

Finally now we can test whether our installation is successful or not. CloudSim includes some test example programs in the 'CloudSim\examples\gridsim\' folder that we can use to test our setup.

1. Compiling a CloudSim program: If you have followed this DIY then compiling a CloudSim program is pretty straightforward; the basic syntax for compilation is just similar to that of Java programs i.e. javac filename.java or javac file_location/filename.java. Let us compile the Example2.java included in 'CloudSim/examples/gridsim/example02/' folder. We will now run the following command in a new command prompt

1	javac /home/dhyan/Desktop/cloudsim-3.0.3/examples/org/cloudbus/cloudsim/examples/CloudSimExample1.java

2. Running the compiled program: The syntax for running a compiled CloudSim program is similar to that of running a program in Java i.e. java filename. In our case we have to type (see image 1)

1	java org.cloudbus.cloudsim.examples.CloudSimExample1

OR if you want to save the output of your program to a file you can use the following

1	java org.cloudbus.cloudsim.examples.CloudSimExample1 > output.txt

Note: The examples given in the CloudSim uses the concept of packages hence it is advisable to go through the basics of packages in Java for a better understanding of the above stated commands.


Image 1. Testing the Setup.


Some Important Points

It is important to set the CLASSPATH correctly, if you make any mistake you might get a class not found error and if you want to supply the classpath externally while compiling or executing using -cp or -classpath; then for some reason it might not work and you would still get the same errors.
Remember to change the 2nd and 3rd CLASSPATH values if you later decide to move the jar files of the CLoudSim or location of your own programs to some other location, respectively and accordingly.
You can also use CloudSim with the Eclipse IDE.
For more information kindly refer the readme.txt and examples.txt provided with the CloudSim.

Goodluck !
