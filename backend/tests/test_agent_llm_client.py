"""Offline protocol, cancellation and bounded retry checks for the model gateway."""
import asyncio
import copy
import json
import re

import httpx
import pytest
from agent.llm import HttpLLMClient, LLMUnavailableError, FixtureLLMClient, build_client


def invoke(client, **kwargs):
    async def run():
        try:
            return await client.complete(system=kwargs.get('system','sys'), messages=kwargs.get('messages',[]), tools=kwargs.get('tools',[]))
        finally:
            await client.aclose()
    return asyncio.run(run())


def client_for(data, capture=None, **kwargs):
    def transport(request):
        if capture is not None:
            capture.append({'url':str(request.url), 'body':json.loads(request.content), 'headers':dict(request.headers)})
        return httpx.Response(200, content=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    return HttpLLMClient(base_url='https://example.com/v1', api_key='test-secret', model='m', transport=httpx.MockTransport(transport), **kwargs)


def test_http_client_payload_and_tool_reply_parsing():
    capture=[]
    reply=invoke(client_for({'choices':[{'message':{'content':'hello','tool_calls':[{'id':'c1','function':{'name':'metrics_lookup','arguments':'{"query":"收益"}'}}]}}], 'usage':{'prompt_tokens':10,'completion_tokens':4}},capture), tools=[{'name':'metrics.lookup','description':'d','parameters':{}}])
    assert reply.content=='hello'
    assert reply.tool_calls[0].name=='metrics.lookup'
    assert reply.tool_calls[0].arguments=={'query':'收益'}
    assert reply.usage['prompt_tokens']==10 and reply.usage['completion_tokens']==4
    assert capture[0]['url']=='https://example.com/v1/chat/completions'
    assert capture[0]['headers']['authorization']=='Bearer test-secret'
    assert capture[0]['body']['tools'][0]['function']['name']=='metrics_lookup'
    assert capture[0]['body']['messages'][0]=={'role':'system','content':'sys'}


def test_tool_names_roundtrip_without_mutating_history():
    specs=[{'name':'metrics.lookup','parameters':{}}, {'name':'metrics.validate','parameters':{}}]
    history=[{'role':'assistant','content':'','tool_calls':[{'id':'c','type':'function','function':{'name':'metrics.lookup','arguments':'{}'}}]}, {'role':'tool','tool_call_id':'c','content':'result'}]
    before=copy.deepcopy((specs,history));capture=[]
    result=invoke(client_for({'choices':[{'message':{'content':'done'}}]},capture),messages=history,tools=specs)
    assert result.content=='done' and (specs,history)==before
    assert capture[0]['body']['messages'][1]['tool_calls'][0]['function']['name']=='metrics_lookup'
    assert capture[0]['body']['messages'][2]['tool_call_id']=='c'
    assert all(re.fullmatch('[A-Za-z0-9_-]{1,64}', t['function']['name']) for t in capture[0]['body']['tools'])


def test_alias_collision_stops_before_network():
    capture=[]
    with pytest.raises(LLMUnavailableError,match='冲突'):
        invoke(client_for({},capture),tools=[{'name':'metrics.lookup'},{'name':'metrics_lookup'}])
    assert capture==[]


@pytest.mark.parametrize(('status','code','count'),[(400,'REQUEST_REJECTED',1),(401,'AUTH_FAILED',1),(403,'AUTH_FAILED',1),(404,'ENDPOINT_NOT_FOUND',1),(429,'RATE_LIMITED',3),(503,'UNAVAILABLE',3)])
def test_http_errors_mask_secrets_and_bound_retries(status,code,count):
    calls=[]
    def transport(req):
        calls.append(req)
        return httpx.Response(status, text='private prompt and private key')
    client=HttpLLMClient(base_url='https://example.com/v1',api_key='private-key',model='m',transport=httpx.MockTransport(transport))
    with pytest.raises(LLMUnavailableError) as err:
        invoke(client,system='private prompt')
    assert err.value.code=='AGENT_LLM_'+code
    assert err.value.upstream_status==status
    assert len(calls)==count
    assert 'private' not in str(err.value)


def test_timeout_and_retry_after_share_one_deadline():
    calls=[]
    def transport(req):
        calls.append(req)
        return httpx.Response(429, headers={'Retry-After':'100'})
    client=HttpLLMClient(base_url='https://example.com/v1',api_key='k',model='m',transport=httpx.MockTransport(transport),timeout_seconds=.02)
    with pytest.raises(LLMUnavailableError) as err:
        invoke(client)
    assert err.value.code=='AGENT_LLM_TIMEOUT' and len(calls)==1


def test_cancellation_propagates_and_does_not_retry():
    async def run():
        entered=asyncio.Event();calls=[]
        async def transport(req):
            calls.append(req);entered.set();await asyncio.Event().wait()
        client=HttpLLMClient(base_url='https://example.com/v1',api_key='k',model='m',transport=httpx.MockTransport(transport))
        task=asyncio.create_task(client.complete(system='s',messages=[],tools=[]))
        await entered.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        await client.aclose()
        assert len(calls)==1
    asyncio.run(run())


@pytest.mark.parametrize('data',[{}, {'choices':[]}, {'choices':[{'message':{'content':None}}]}, {'choices':[{'message':{'content':['wrong']}}]}, {'choices':[{'message':{'tool_calls':[{'id':'c','function':{'name':'x','arguments':'[]'}}]}}]}, {'choices':[{'message':{'tool_calls':[{'id':'c','function':{'name':'x','arguments':'{'}}]}}]}, {'choices':[{'message':{'tool_calls':[{'id':'c','function':{'name':'x'}},{'id':'c','function':{'name':'x'}}]}}]}])
def test_malformed_replies_are_typed_and_not_retried(data):
    capture=[]
    with pytest.raises(LLMUnavailableError) as err:invoke(client_for(data,capture))
    assert err.value.code=='AGENT_LLM_INVALID_RESPONSE' and len(capture)==1


def test_body_size_is_bounded(monkeypatch):
    from agent import llm
    monkeypatch.setattr(llm,'MAX_RESPONSE_BYTES',100)
    with pytest.raises(LLMUnavailableError,match='过大'):
        invoke(client_for({'choices':[{'message':{'content':'x'*200}}]}))


def test_fixture_and_unknown_usage():
    reply=invoke(client_for({'choices':[{'message':{'content':'ok'}}]}))
    assert reply.usage['prompt_tokens'] is None
    fixture=FixtureLLMClient([{'content':'ok'}])
    assert invoke(fixture).content=='ok'
    assert build_client({}) is None
    assert build_client({'api_key':' '}) is None
    assert build_client({'api_key':'test','enabled':False}) is None
    assert isinstance(build_client({'api_key':'test'}),HttpLLMClient)


@pytest.mark.parametrize('message,finish',[({'content':'partial'},'length'), ({'content':'blocked'},'content_filter'), ({'tool_calls':[{'id':'c','function':{'name':'x','arguments':'{"window":NaN}'}}]},'tool_calls'), ({'tool_calls':[{'id':'c','function':{'name':'x','arguments':'{"window":1e1000}'}}]},'tool_calls')])
def test_incomplete_or_nonfinite_model_output_never_reaches_tools(message,finish):
    with pytest.raises(LLMUnavailableError) as err:
        invoke(client_for({'choices':[{'message':message,'finish_reason':finish}]}))
    assert err.value.code=='AGENT_LLM_INVALID_RESPONSE'


@pytest.mark.parametrize('effort', ['default', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
def test_reasoning_effort_is_explicit_and_does_not_force_sampling(effort):
    capture = []
    invoke(client_for({'choices': [{'message': {'content': 'ok'}}]}, capture, reasoning_effort=effort))
    body = capture[0]['body']
    assert 'thinking' not in body
    if effort == 'default':
        assert 'reasoning_effort' not in body and body['temperature'] == 0
    else:
        assert body['reasoning_effort'] == effort and 'temperature' not in body
    assert 'x-opencode-session' not in capture[0]['headers']


@pytest.mark.parametrize('effort', ['default', 'none', 'max'])
def test_deepseek_go_headers_thinking_and_reasoning_roundtrip(effort):
    requests = []
    def transport(req):
        requests.append(req)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'ok', 'reasoning_content': 'fixture reasoning'}}]})
    client = HttpLLMClient(base_url='https://opencode.ai/zen/go/v1', api_key='test', model='deepseek-v4.1-flash',
                          reasoning_effort=effort, session_id='session-123', transport=httpx.MockTransport(transport))
    history = [{'role': 'assistant', 'content': '历史回复'}]
    result = invoke(client, messages=history)
    assert result.reasoning_content == 'fixture reasoning'
    body = json.loads(requests[0].content)
    assert body['messages'][1]['reasoning_content'] == ''
    assert 'reasoning_content' not in history[0]
    assert requests[0].headers['x-opencode-session'] == 'session-123'
    assert requests[0].headers['user-agent'] == 'AiFunctions/1.0'
    if effort == 'default':
        assert 'thinking' not in body and 'reasoning_effort' not in body
    else:
        assert body['thinking']['type'] == ('disabled' if effort == 'none' else 'enabled')
        assert body['reasoning_effort'] == effort


def test_go_missing_session_and_invalid_reasoning_fail_closed():
    client = HttpLLMClient(base_url='https://opencode.ai/zen/go/v1', api_key='test', model='deepseek-v4.1-flash',
                          transport=httpx.MockTransport(lambda req: pytest.fail('must not call upstream')))
    with pytest.raises(LLMUnavailableError, match='会话标识'):
        invoke(client)
    for invalid_reasoning in [{'raw': 'bad'}, ['bad'], 5, '\ud800']:
        with pytest.raises(LLMUnavailableError):
            invoke(client_for({'choices': [{'message': {'content': 'ok', 'reasoning_content': invalid_reasoning}}]}))


@pytest.mark.parametrize('status,body,expected', [
    (400, {'error': {'code':'context_length_exceeded','message':'private upstream detail'}}, 'AGENT_LLM_CONTEXT_OVERFLOW'),
    (422, {'error': {'message':'Input token count exceeds model limit'}}, 'AGENT_LLM_CONTEXT_OVERFLOW'),
    (400, {'error': {'message':'unsupported reasoning_effort'}}, 'AGENT_LLM_REQUEST_REJECTED'),
    (401, {'error': {'code':'context_length_exceeded'}}, 'AGENT_LLM_AUTH_FAILED'),
    (413, {'error': {'message':'request body too large'}}, 'AGENT_LLM_REQUEST_REJECTED'),
])
def test_classifies_only_explicit_input_overflow(status, body, expected):
    attempts=[]
    def handler(request):
        attempts.append(1)
        return httpx.Response(status,json=body)
    client=HttpLLMClient(base_url='https://example.com/v1',model='m',api_key='test',transport=httpx.MockTransport(handler))
    with pytest.raises(LLMUnavailableError) as err: invoke(client)
    assert err.value.code==expected and len(attempts)==1
    assert 'private upstream detail' not in str(err.value)


def test_error_body_is_bounded_without_echoing_raw_content():
    class Body(httpx.AsyncByteStream):
        reads=0
        async def __aiter__(self):
            for _ in range(100):
                self.reads+=1
                yield b'private-error-content'*1000
    body=Body()
    client=HttpLLMClient(base_url='https://example.com/v1',model='m',api_key='test',
                        transport=httpx.MockTransport(lambda req:httpx.Response(400,stream=body)))
    with pytest.raises(LLMUnavailableError) as err: invoke(client)
    assert body.reads==1 and 'private-error' not in str(err.value)


def test_build_client_preserves_explicit_capacity():
    client=build_client({'base_url':'https://example.com/v1','api_key':'fixture','model':'m','context_window_tokens':65536})
    assert client.context_window_tokens==65536
