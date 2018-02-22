from nlpcore import*

with open('./downloads/eval_annotated.xml') as fd:
    ETree = ET.parse(fd)

i = 0
for ele in ETree.getroot().iter():
    if ele.tag == 'sentence':
        i += 1
        if ele.get('negation','abc') != 'abc':
            continue
        print('No. '+str(i)+': '+toString(ele))
        neg = input("Negation? (y/[n])")
        if neg == 'y':
            ele.set('negation','true')
        elif neg == 'exit':
            break
        else:
            ele.set('negation','false')

print('Finished annotations!')
ETree.write('./downloads/eval_annotated.xml')
