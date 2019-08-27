/*
 * Test du capteur MPX5100 AP.
 * 
 * Auteurs :
 * Marc-Antoine BUCHET
 * Hugues Limousin
 * 
 * Le but est de tester le capteur.
 * Testé avec un Arduino Uno.
 */

// Déclaration des pins :
int mesurePin = 5 ;

// Déclaration des variables :
int nbreCycles = 5 ; // Pour moyenner les mesures
int tempsAttente  = 100 ; // Ici, c'est environ la durée (en ms) entre deux mesures lors du moyennage
int mesure = 0 ; // valeur mesurée entre 0 et 1023
float pression = 0.0 ; // pression en Pa (obtenue par conversion de msure via une calibration) 
float sommePression = 0.0 ; // Pour calcul de la pression moyenne
float pressionMoyenne = 0.0 ; 

void setup() {
Serial.begin(9600); // set the baud rate
Serial.println("Ready") ; // print "Ready" once
}
void loop() {
  sommePression = 0.0 ;
  for(int i=0 ; i<nbreCycles ; i++) {
    mesure = analogRead(mesurePin) ;
    pression = (mesure * 5./1023.)*223.706+110.351 ; // Utiliser la calibration à terme
    sommePression += pression ;
    //Serial.println(mesure) ;
    delay(tempsAttente) ;
  }
  pressionMoyenne = sommePression/nbreCycles ;
  Serial.println(pressionMoyenne) ;
  delay(tempsAttente) ;
}
