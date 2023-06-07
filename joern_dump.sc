@main def exec(cpgFile: String) = {
   importCpg(cpgFile)
   run.dumpcpg14
}