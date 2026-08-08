Verstanden. Du brauchst kein formales Anforderungsdokument, sondern einen kompakten gedanklichen Bauplan, an dem du dich während der Umsetzung orientieren kannst. Der Fokus liegt deshalb auf Datenstrukturen, Reihenfolge, Ableitungen und minimalem Pseudocode.

Gedanklicher Leitfaden für ein neuronales Netz
1. Das Gesamtbild

Ein einfaches neuronales Netz kann gedanklich in vier Bereiche aufgeteilt werden:

1. Netzwerk
   Enthält die Schichten

2. Schichten
   Verarbeiten Werte im Forward Pass
   Verarbeiten Gradienten im Backward Pass

3. Loss-Funktion
   Bewertet die Vorhersage

4. Optimierer
   Aktualisiert Gewichte und Bias-Werte


Der gesamte Trainingsablauf ist:

Eingabe
  ↓
Forward Pass
  ↓
Vorhersage
  ↓
Loss berechnen
  ↓
Backward Pass
  ↓
Gradienten
  ↓
Parameter aktualisieren

2. Wichtige Datentypen

Für ein kleines neuronales Netz reichen gedanklich zunächst folgende Datentypen.

Skalar

Eine einzelne Zahl.

Beispiele:

Loss
Lernrate
ein einzelnes Gewicht


Mathematisch:

L∈RL\in\mathbb{R}
Vektor

Eine geordnete Liste von Zahlen.

Beispiele:

Eingabe x
Bias b
Aktivierungen h
Output y
Fehlersignal δ


Beispiel:

x=(x1x2x3)x= \begin{pmatrix} x_1\\ x_2\\ x_3 \end{pmatrix}
Matrix

Eine rechteckige Anordnung von Zahlen.

Beispiele:

Gewichtsmatrix W
Gewichtsgradient dW
mehrere Trainingsbeispiele


Beispiel:

W=(w11w12w13w21w22w23)W= \begin{pmatrix} w_{11}&w_{12}&w_{13}\\ w_{21}&w_{22}&w_{23} \end{pmatrix}
3. Aufbau einer Schicht

Jede Schicht sollte gedanklich drei Bereiche besitzen:

Schicht
├── Parameter
├── gespeicherte Forward-Werte
└── berechnete Gradienten

Parameter

Werte, die das Netz lernt:

W = Gewichtsmatrix
b = Bias-Vektor

Gespeicherte Forward-Werte

Werte, die später beim Backward Pass benötigt werden:

Eingabe der Schicht
eventuell Ausgabe vor der Aktivierungsfunktion

Gradienten

Ergebnisse des Backward Passes:

dW = Gradient der Gewichte
db = Gradient der Bias-Werte

4. Dense-Schicht

Eine Dense-Schicht berechnet:

z=Wa+bz=Wa+b

Dabei ist:

a = Eingabe der Schicht
W = Gewichtsmatrix
b = Bias-Vektor
z = Ausgabe der Schicht

Gedankliche Struktur
Dense-Schicht

Parameter:
- W
- b

Gespeichert:
- a

Gradienten:
- dW
- db

Operationen:
- forward(a)
- backward(dz)

Warum wird die Eingabe gespeichert?

Der Gewichtsgradient wird später mit dieser Formel berechnet:

dW=dz⋅aTdW=dz\cdot a^T

Die Dense-Schicht muss daher wissen, welche Eingabe im Forward Pass verwendet wurde.

5. Aktivierungsschicht

Die ReLU-Aktivierung berechnet:

h=ReLU⁡(z)h=\operatorname{ReLU}(z)

mit:

ReLU⁡(z)=max⁡(0,z)\operatorname{ReLU}(z)=\max(0,z)
Gedankliche Struktur
ReLU-Schicht

Parameter:
- keine

Gespeichert:
- z

Operationen:
- forward(z)
- backward(dh)


ReLU speichert zz, weil im Backward Pass geprüft werden muss, welche Werte positiv waren.

Die Ableitung ist:

ReLU⁡′(z)={1,z>00,z≤0\operatorname{ReLU}'(z)= \begin{cases} 1,&z>0\\ 0,&z\leq0 \end{cases}

Der Backward-Schritt lautet:

dz=dh⊙ReLU⁡′(z)dz=dh\odot\operatorname{ReLU}'(z)

Das Symbol ⊙\odot bedeutet elementweise Multiplikation.

6. Beispielstruktur des Netzes

Wir verwenden:

Input
  ↓
Dense 1
  ↓
ReLU
  ↓
Dense 2
  ↓
Output-Aktivierung
  ↓
Loss


Mathematisch:

z1=W1x+b1z_1=W_1x+b_1 h=ReLU⁡(z1)h=\operatorname{ReLU}(z_1) z2=W2h+b2z_2=W_2h+b_2 y=g(z2)y=g(z_2) L=Loss⁡(y,t)L=\operatorname{Loss}(y,t)

Dabei ist:

x = Eingabe
t = gewünschtes Ergebnis
y = Vorhersage
L = Fehler

7. Forward Pass
Aufgabe

Der Forward Pass berechnet aus einer Eingabe eine Vorhersage und anschließend den Loss.

Reihenfolge
1. Eingabe x annehmen

2. Hidden Dense:
   z₁ = W₁x + b₁

3. ReLU:
   h = ReLU(z₁)

4. Output Dense:
   z₂ = W₂h + b₂

5. Output-Aktivierung:
   y = g(z₂)

6. Loss:
   L = Loss(y, t)

Was muss gespeichert werden?
Dense 1 speichert:
- x

ReLU speichert:
- z₁

Dense 2 speichert:
- h

Output/Loss speichern abhängig von der Funktion:
- z₂
- y
- t

Minimaler Pseudocode
forward(x, t):

    z₁ = dense1.forward(x)
    h  = relu.forward(z₁)
    z₂ = dense2.forward(h)
    y  = outputActivation.forward(z₂)
    L  = loss.forward(y, t)

    return y, L


Der Pseudocode zeigt nur den Ablauf. Intern speichern die einzelnen Komponenten die Werte, die später benötigt werden.

8. Backward Pass
Aufgabe

Der Backward Pass beantwortet:

Wie stark beeinflusst jeder Parameter den Loss?

Der Backward Pass läuft in genau umgekehrter Reihenfolge:

Forward:
x → z₁ → h → z₂ → y → L

Backward:
L → y → z₂ → h → z₁ → x


Jede Schicht erhält einen Gradienten und gibt einen neuen Gradienten nach hinten weiter.

9. Prinzip der Kettenregel

Angenommen:

w→z→y→Lw\rightarrow z\rightarrow y\rightarrow L

Dann beeinflusst ww den Loss nicht direkt, sondern über mehrere Zwischenschritte.

Deshalb gilt:

∂L∂w=∂L∂y⋅∂y∂z⋅∂z∂w\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}

Programmatisch muss diese Formel nicht vollständig auf einmal berechnet werden.

Stattdessen reicht jede Komponente ihren Teil weiter:

Loss liefert Gradient an Output
Output liefert Gradient an Dense 2
Dense 2 liefert Gradient an ReLU
ReLU liefert Gradient an Dense 1

10. Backward Pass einer Dense-Schicht

Die Dense-Schicht berechnete im Forward Pass:

z=Wa+bz=Wa+b

Im Backward Pass erhält die Schicht:

dz=∂L∂zdz=\frac{\partial L}{\partial z}

Aus diesem Gradienten berechnet sie drei Dinge.

Gewichtsgradient
dW=dz⋅aT\boxed{dW=dz\cdot a^T}

Bedeutung:

Gewichtsgradient
=
Fehler am Zielneuron
×
Eingabe des Gewichts

Bias-Gradient
db=dz\boxed{db=dz}

Bei mehreren Trainingsbeispielen werden die Bias-Gradienten über die Batch-Dimension zusammengefasst.

Gradient für die vorherige Schicht
da=WT⋅dz\boxed{da=W^T\cdot dz}

Dieser Gradient wird an die vorherige Schicht zurückgegeben.

Minimaler Pseudocode
Dense.backward(dz):

    dW = dz × gespeicherteEingabeᵀ
    db = dz
    da = Wᵀ × dz

    speichere dW
    speichere db

    return da


Wichtig: Für dada werden die noch nicht aktualisierten Gewichte benötigt.

11. Backward Pass der ReLU-Schicht

Die ReLU-Schicht erhält den Gradienten dhdh.

Sie verwendet die gespeicherten Werte zz:

dz=dh⊙ReLU⁡′(z)dz=dh\odot\operatorname{ReLU}'(z)
Bedeutung
Wenn z > 0:
Gradient weitergeben

Wenn z ≤ 0:
Gradient auf 0 setzen

Minimaler Pseudocode
ReLU.backward(dh):

    Maske = 1 für gespeichertes z > 0
            0 für gespeichertes z ≤ 0

    dz = dh elementweise multipliziert mit Maske

    return dz


ReLU besitzt keine Gewichte. Deshalb entstehen hier weder dWdW noch dbdb.

12. Vollständiger Backward Pass

Der Backward Pass beginnt beim Loss.

Reihenfolge
1. Loss rückwärts
2. Output-Aktivierung rückwärts
3. Dense 2 rückwärts
4. ReLU rückwärts
5. Dense 1 rückwärts

Mathematische Übersicht

Ausgangsgradient:

δ2=∂L∂z2\delta_2=\frac{\partial L}{\partial z_2}

Output Layer:

dW2=δ2hTdW_2=\delta_2h^T db2=δ2db_2=\delta_2 dh=W2Tδ2dh=W_2^T\delta_2

ReLU:

δ1=dh⊙ReLU⁡′(z1)\delta_1=dh\odot\operatorname{ReLU}'(z_1)

Hidden Layer:

dW1=δ1xTdW_1=\delta_1x^T db1=δ1db_1=\delta_1

Optional:

dx=W1Tδ1dx=W_1^T\delta_1
Minimaler Pseudocode
backward():

    dy  = loss.backward()
    dz₂ = outputActivation.backward(dy)

    dh  = dense2.backward(dz₂)
    dz₁ = relu.backward(dh)
    dx  = dense1.backward(dz₁)

    return dx


Bei einer passenden Kombination aus Softmax und Cross-Entropy können Loss und Output-Aktivierung gemeinsam behandelt werden:

δ2=y−t\delta_2=y-t

Dann entfällt eine separate Berechnung von dydy.

13. Parameter aktualisieren

Der Backward Pass berechnet nur:

dW₁
db₁
dW₂
db₂


Er verändert noch keine Parameter.

Erst der Optimierer aktualisiert die Parameter:

W←W−ηdWW\leftarrow W-\eta dW b←b−ηdbb\leftarrow b-\eta db
Minimaler Pseudocode
update():

    dense1.W = dense1.W - lernrate × dense1.dW
    dense1.b = dense1.b - lernrate × dense1.db

    dense2.W = dense2.W - lernrate × dense2.dW
    dense2.b = dense2.b - lernrate × dense2.db

Wichtige Reihenfolge
1. Alle Gradienten berechnen
2. Erst danach alle Parameter aktualisieren


Nicht:

Gradient einer Schicht berechnen
Parameter sofort verändern
Backward Pass mit veränderten Parametern fortsetzen

14. Gesamter Trainingsschritt
trainStep(x, t):

    # Forward
    y, L = forward(x, t)

    # Backward
    backward()

    # Update
    update()

    # Gradienten für nächsten Schritt zurücksetzen
    resetGradients()

    return y, L


Gedanklich:

Daten
  ↓
Vorhersage
  ↓
Fehler
  ↓
Gradienten
  ↓
Parameter-Update

15. Netzwerk als Sammlung von Schichten

Anstatt jede Schicht einzeln fest einzubauen, kann das Netzwerk gedanklich eine geordnete Liste von Schichten besitzen:

Schichten:

1. Dense
2. ReLU
3. Dense
4. Output-Aktivierung

Forward Pass

Die Schichten werden von vorne nach hinten durchlaufen:

Wert = Eingabe

für jede Schicht von vorne nach hinten:
    Wert = Schicht.forward(Wert)

Vorhersage = Wert

Backward Pass

Die Schichten werden in umgekehrter Reihenfolge durchlaufen:

Gradient = Loss-Gradient

für jede Schicht von hinten nach vorne:
    Gradient = Schicht.backward(Gradient)


Das ist eine besonders wichtige Strukturidee:

Forward:
erste Schicht → letzte Schicht

Backward:
letzte Schicht → erste Schicht

16. Dimensionen als Kontrollwerkzeug

Dimensionen helfen dir, Fehler sofort zu erkennen.

Für:

3 Inputs
4 Hidden-Neuronen
2 Outputs


gelten:

x    : 3 × 1
W₁   : 4 × 3
b₁   : 4 × 1
z₁   : 4 × 1
h    : 4 × 1

W₂   : 2 × 4
b₂   : 2 × 1
z₂   : 2 × 1
y    : 2 × 1


Backward:

δ₂   : 2 × 1
dW₂  : 2 × 4
db₂  : 2 × 1
dh   : 4 × 1

δ₁   : 4 × 1
dW₁  : 4 × 3
db₁  : 4 × 1
dx   : 3 × 1

Wichtigste Dimensionsregeln

Ein Parameter und sein Gradient müssen dieselbe Form besitzen:

Form⁡(dW)=Form⁡(W)\operatorname{Form}(dW)=\operatorname{Form}(W) Form⁡(db)=Form⁡(b)\operatorname{Form}(db)=\operatorname{Form}(b)

Wenn das nicht stimmt, ist meistens:

die Reihenfolge einer Multiplikation falsch,
eine Transposition vergessen worden,
oder die Schichtdimension wurde falsch festgelegt.
17. Welche Werte existieren wie lange?
Dauerhaft

Diese Werte gehören zum trainierten Modell:

W₁
b₁
W₂
b₂

Nur für einen Forward- und Backward-Durchlauf
x
z₁
h
z₂
y
t


Nach dem Backward Pass werden diese Werte für den aktuellen Durchlauf nicht mehr benötigt.

Gradienten
dW₁
db₁
dW₂
db₂


Sie werden im Backward Pass erzeugt und vom Optimierer verwendet.

18. Initialisierung

Vor dem ersten Training müssen die Parameter initialisiert werden.

Gewichte

Gewichte sollten:

kleine Werte besitzen,
voneinander verschieden sein,
nicht alle auf null gesetzt werden.

Wenn alle Gewichte gleich sind, lernen die Neuronen dasselbe.

Bias-Werte

Bias-Werte können für ein erstes einfaches Netz mit null beginnen.

19. Training und Vorhersage unterscheiden
Training
Forward
Loss
Backward
Update

Vorhersage
nur Forward


Bei einer Vorhersage werden:

keine Zielwerte benötigt,
keine Gradienten berechnet,
keine Parameter verändert.
20. Minimale Komponenten

Für eine erste eigene Umsetzung reichen diese Komponenten:

1. Matrix- und Vektoroperationen

2. Dense-Schicht
   - forward
   - backward
   - W, b, dW, db

3. ReLU
   - forward
   - backward

4. Output-Aktivierung
   - abhängig von der Aufgabe

5. Loss
   - forward
   - backward

6. Netzwerk
   - verwaltet Schichten
   - führt Forward und Backward aus

7. Optimierer
   - aktualisiert Parameter

8. Trainingsschleife
   - wiederholt alle Schritte

21. Empfohlene Reihenfolge der Umsetzung
Schritt 1:
Vektor- und Matrixdimensionen festlegen

Schritt 2:
Dense Forward umsetzen

Schritt 3:
ReLU Forward umsetzen

Schritt 4:
Gesamten Forward Pass testen

Schritt 5:
Loss berechnen

Schritt 6:
Loss-Ableitung bestimmen

Schritt 7:
Dense Backward umsetzen

Schritt 8:
ReLU Backward umsetzen

Schritt 9:
Gesamten Backward Pass testen

Schritt 10:
Parameter-Update ergänzen

Schritt 11:
Trainingsschleife ergänzen

Schritt 12:
Gradienten numerisch überprüfen

22. Kompakter Spickzettel
Dense Forward
z=Wa+bz=Wa+b

Speichern:

aa
Dense Backward
dW=dz⋅aTdW=dz\cdot a^T db=dzdb=dz da=WTdzda=W^Tdz
ReLU Forward
h=max⁡(0,z)h=\max(0,z)

Speichern:

zz
ReLU Backward
dz=dh⊙ReLU⁡′(z)dz=dh\odot\operatorname{ReLU}'(z)
Forward des gesamten Netzes
z1=W1x+b1z_1=W_1x+b_1 h=ReLU⁡(z1)h=\operatorname{ReLU}(z_1) z2=W2h+b2z_2=W_2h+b_2 y=g(z2)y=g(z_2) L=Loss⁡(y,t)L=\operatorname{Loss}(y,t)
Backward des gesamten Netzes
δ2=∂L∂z2\delta_2=\frac{\partial L}{\partial z_2} dW2=δ2hTdW_2=\delta_2h^T db2=δ2db_2=\delta_2 dh=W2Tδ2dh=W_2^T\delta_2 δ1=dh⊙ReLU⁡′(z1)\delta_1=dh\odot\operatorname{ReLU}'(z_1) dW1=δ1xTdW_1=\delta_1x^T db1=δ1db_1=\delta_1
Update
W←W−ηdWW\leftarrow W-\eta dW b←b−ηdbb\leftarrow b-\eta db
23. Das wichtigste mentale Modell
FORWARD

Eine Schicht erhält Werte.
Eine Schicht berechnet neue Werte.
Eine Schicht speichert notwendige Zwischenwerte.
Eine Schicht gibt das Ergebnis weiter.


BACKWARD

Eine Schicht erhält einen Gradienten.
Eine trainierbare Schicht berechnet dW und db.
Eine Schicht berechnet den Gradienten ihrer Eingabe.
Eine Schicht gibt diesen Gradienten nach hinten weiter.


UPDATE

Der Optimierer nimmt dW und db.
Der Optimierer verändert W und b.


Der wichtigste Leitsatz lautet:

Im Forward Pass fließen Werte nach vorne. Im Backward Pass fließen Ableitungen nach hinten. Jede Schicht kennt nur ihre eigene Berechnung und muss deshalb nur deren lokale Ableitung beherrschen.