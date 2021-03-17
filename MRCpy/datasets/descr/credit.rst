
1. Title: Credit Approval

2. Sources: 
    (confidential)
    Submitted by quinlan@cs.su.oz.au

3.  Past Usage:

    See Quinlan,
    * "Simplifying decision trees", Int J Man-Machine Studies 27,
      Dec 1987, pp. 221-234.
    * "C4.5: Programs for Machine Learning", Morgan Kaufmann, Oct 1992
  
4.  Relevant Information:

    This file concerns credit card applications.  All attribute names
    and values have been changed to meaningless symbols to protect
    confidentiality of the data.
  
    This dataset is interesting because there is a good mix of
    attributes -- continuous, nominal with small numbers of
    values, and nominal with larger numbers of values.  There
    are also a few missing values.
  
5.  Number of Instances: 690

6.  Number of Attributes: 15 + class attribute

7.  Attribute Information:

    A1:	b, a.
    A2:	continuous.
    A3:	continuous.
    A4:	u, y, l, t.
    A5:	g, p, gg.
    A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    A7:	v, h, bb, j, n, z, dd, ff, o.
    A8:	continuous.
    A9:	t, f.
    A10: t, f.
    A11: continuous.
    A12: t, f.
    A13: g, p, s.
    A14: continuous.
    A15: continuous.
    A16: +,- (class attribute)

8.  Missing Attribute Values:
    37 cases (5%) have one or more missing values.  The missing
    values from particular attributes are:

    A1:  12
    A2:  12
    A4:   6
    A5:   6
    A6:   9
    A7:   9
    A14: 13
    
9.  Class Distribution
  
    +: 307 (44.5%)
    -: 383 (55.5%)


We have transform the UCI dataset for classification purpose.
1- Replace the missing value '?' by 0
2- Mapped the categorical attribute's values to integer using the following dict mapping:

{'A1': {0: '0', 1: 'a', 2: 'b'},
 'A10': {0: '0', 1: 't'},
 'A12': {0: '0', 1: 't'},
 'A13': {0: 'g', 1: 'p', 2: 's'},
 'A16': {0: '+', 1: '-'},
 'A4': {0: '0', 1: 'l', 2: 'u', 3: 'y'},
 'A5': {0: '0', 1: 'g', 2: 'gg', 3: 'p'},
 'A6': {0: '0', 1: 'aa', 2: 'c', 3: 'cc', 4: 'd', 5: 'e', 6: 'ff', 7: 'i', 8: 'j', 9: 'k', 10: 'm', 11: 'q', 12: 'r', 13: 'w', 14: 'x'},
 'A7': {0: '0', 1: 'bb', 2: 'dd', 3: 'ff', 4: 'h', 5: 'j', 6: 'n', 7: 'o', 8: 'v', 9: 'z'},
 'A9': {0: '0', 1: 't'}}
 'A12': {0: 'f', 1: 't'},
 'A13': {0: 'g', 1: 'p', 2: 's'}}
 
