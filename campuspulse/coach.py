"""Retrieve curated suggestions; constrain local LLM to approved authored text."""
import json
import os
import re
import urllib.request
from .data import validate_input
from .contracts import validate_schema_input
from .privacy import LOCAL_CHAT_URL, open_local_llm
from .knowledge import CARDS, BY_ID, approved_wordings, retrieve

FOCUS = {
    'general':('A general suggestion','connection routine break'),
    'connection':('Connection with others','social friends connection community'),
    'routine':('Sleep and routine','sleep bedtime routine rest'),
    'movement':('Movement','activity exercise movement walk'),
    'balance':('Study and screen breaks','screen study balance break'),
}


def normalized(text):
    return ' '.join(text.split()).casefold()


def support(payload,consent=False,schema=None):
    if consent is not True:
        return {'mode':'disabled','message':'Support suggestions are off until you opt in.','sources':[]}
    values=validate_schema_input(payload,schema) if schema else validate_input(payload)
    focus=payload.get('weekly_focus','general')
    if not isinstance(focus,str) or focus not in FOCUS:
        raise ValueError('Choose a supported weekly focus.')
    query=payload.get('support_query','')
    if not isinstance(query,str) or len(query)>240:
        raise ValueError('Use a support query of at most 240 characters.')
    query=query.strip()
    if re.search(r'diagnos|medicat|dosage|treatment|suicid|self[- ]harm',query,re.I):
        return {'mode':'out_of_scope','message':'This prototype only covers general routines, connection, movement and study breaks. This request is outside its scope.','sources':[],
            'note':'No LLM request was made.','reason':'The query requests support outside the curated scope.'}
    topics=[]
    if query:
        sources=retrieve(query,None if focus=='general' else focus)
        if not sources:
            return {'mode':'no_match','message':'No matching support material was found. Try a specific routine or study topic, or clear the search for a general suggestion.','sources':[],
                'note':'No LLM request was made.','reason':'No lexical match in the curated collection.'}
        topics=list(dict.fromkeys(c['topic'] for c in sources))
    elif focus!='general':
        sources=[{**BY_ID[focus],'similarity':1.0}]
        topics=[focus]
    elif schema:
        sources=[{**BY_ID['connection'],'similarity':1.0}]
        topics=['connection']
    else:
        if values['social_minutes']<45:topics.append('social connection friends')
        if values['sleep_hours']<6 or values['routine_variability']>1.8:topics.append('sleep bedtime routine')
        if values['active_minutes']<25:topics.append('activity movement')
        if values['screen_hours']>6:topics.append('screen break balance')
        sources=retrieve(' '.join(topics) or 'connection routine break')
    reason=('Retrieved from your optional search within the chosen topic.' if query else
            f'You chose {FOCUS[focus][0].lower()}.' if focus!='general' else 'You requested a general suggestion.')
    reason+=' Your predicted score does not select this suggestion.'
    fallback=sources[0]['text']

    def result(mode,message,note,selected=None,failure=None):
        ordered=sorted(sources,key=lambda c:c['id']!=selected) if selected else sources
        return {'mode':mode,'message':message,'sources':ordered,'note':note,'reason':reason,
            'weekly_focus':focus,'selected_card_id':selected or sources[0]['id'],'fallback_reason':failure}

    model=os.getenv('OLLAMA_MODEL','').strip()
    if not model:
        return result('retrieval',fallback,'Offline TF-IDF retrieval. Text is authored; no LLM was called.')
    # The raw query stays in local lexical retrieval. Only generalized topic
    # names and retrieved approved wording go to the separate local service.
    options=[{'id':c['id'],'text':text} for c in sources for text in approved_wordings(c)]
    allowed={normalized(o['text']):(o['text'],o['id']) for o in options}
    body={'model':model,'stream':False,'messages':[
        {'role':'system','content':'Choose one provided suggestion suitable for the topics. Return its text EXACTLY, with no heading, extra words, explanation or follow-up. Do not rewrite or invent content. All displayed wording must come from the provided choices.'},
        {'role':'user','content':json.dumps({'topics':topics,'cards':options})}],
        'options':{'temperature':0,'num_predict':100}}
    try:
        request=urllib.request.Request(LOCAL_CHAT_URL,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with open_local_llm(request,timeout=20) as response:
            generated=json.load(response)['message']['content']
        if not isinstance(generated,str) or normalized(generated) not in allowed:
            return result('retrieval_fallback',fallback,'Output rejected: it was not an approved complete suggestion. Showing authored text.',failure='unapproved_text')
        canonical,selected=allowed[normalized(generated)]
        return result('local_llm',canonical,'Local LLM selected approved wording. Displayed text is authored, not a free-form AI health assessment.',selected)
    except Exception:
        return result('retrieval_fallback',fallback,'Local LLM request failed or returned an unexpected response. Showing authored text.',failure='request_or_response_error')
