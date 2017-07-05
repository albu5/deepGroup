function varargout = pairwise(varargin)
% PAIRWISE MATLAB code for pairwise.fig
%      PAIRWISE is an annotation tool for piarwise interaction in collective
% 	   activity dataset. However, the design can be extended to other annotatios.
%      
%      To get started, just run pairwise.m and fill the sequence number to be
%      annotated in the text box. Press start and start annotating.
%      
%      To edit existing annotation, change flag=false in function start_Callback
%      
%      PAIRWISE, by itself, creates a new PAIRWISE or raises the existing
%      singleton*.
%
%      H = PAIRWISE returns the handle to a new PAIRWISE or the handle to
%      the existing singleton*.
%
%      PAIRWISE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PAIRWISE.M with the given input arguments.
%
%      PAIRWISE('Property','Value',...) creates a new PAIRWISE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pairwise_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pairwise_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pairwise

% Last Modified by GUIDE v2.5 27-Jun-2017 02:26:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @pairwise_OpeningFcn, ...
                   'gui_OutputFcn',  @pairwise_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before pairwise is made visible.
function pairwise_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to pairwise (see VARARGIN)

% Choose default command line output for pairwise
handles.output = hObject;

% Update handles structure

handles.data_dir = './ActivityDataset';
handles.anno_dir = fullfile(handles.data_dir, 'annotations');
handles.seq_str = 'seq%2.2d';
handles.anno_str = 'anno%2.2d.mat';
handles.im_str = 'frame%4.4d.jpg';
handles.interact_labels = {'no-interaction'...
    'approaching',...
    'leaving',...
    'passing-by',...
    'facing-each-other',...
    'walking-together',...
    'standing-in-a-row',...
    'standing-side-by-side',...
    'undefined'
    };
guidata(hObject, handles)

% This sets up the initial plot - only do when we are invisible
% so window can get raised using pairwise.
if strcmp(get(hObject,'Visible'),'off')
    plot(rand(5));
end

% UIWAIT makes pairwise wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = pairwise_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in nointer.
function nointer_Callback(hObject, eventdata, handles)
% hObject    handle to nointer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = 1;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);


% --- Executes on button press in approach.
function approach_Callback(hObject, eventdata, handles)
% hObject    handle to approach (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = 2;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);



% --- Executes on button press in leaving.
function leaving_Callback(hObject, eventdata, handles)
% hObject    handle to leaving (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = 3;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);



% --- Executes on button press in passing.
function passing_Callback(hObject, eventdata, handles)
% hObject    handle to passing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = 4;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);



% --- Executes on button press in facing.
function facing_Callback(hObject, eventdata, handles)
% hObject    handle to facing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = 5;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);


% --- Executes on button press in walking.
function walking_Callback(hObject, eventdata, handles)
% hObject    handle to walking (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = 6;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);



% --- Executes on button press in row.
function row_Callback(hObject, eventdata, handles)
% hObject    handle to row (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = 7;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);



% --- Executes on button press in side.
function side_Callback(hObject, eventdata, handles)
% hObject    handle to side (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = 8;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);


% --- Executes on button press in same.
function same_Callback(hObject, eventdata, handles)
% hObject    handle to same (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = handles.current.label;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);



function seqn_Callback(hObject, eventdata, handles)
% hObject    handle to seqn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.data_dir = './ActivityDataset';
handles.anno_dir = fullfile(handles.data_dir, 'annotations');
handles.seq_str = 'seq%2.2d';
handles.anno_str = 'anno%2.2d.mat';
handles.im_str = 'frame%4.4d.jpg';
handles.interact_labels = {'no-interaction'...
    'approaching',...
    'leaving',...
    'passing-by',...
    'facing-each-other',...
    'walking-together',...
    'standing-in-a-row',...
    'standing-side-by-side',...
    'undefined'
    };

% Hints: get(hObject,'String') returns contents of seqn as text
%        str2double(get(hObject,'String')) returns contents of seqn as a double
handles.seqn = str2double(get(hObject, 'String'));
handles.im_dir = fullfile(handles.data_dir, sprintf(handles.seq_str, handles.seqn));
handles.anno_file = fullfile(handles.anno_dir, sprintf(handles.anno_str, handles.seqn));
handles.framen = 1;
guidata(hObject, handles)
display(handles.im_dir)
display(handles.anno_file)


% --- Executes during object creation, after setting all properties.
function seqn_CreateFcn(hObject, eventdata, handles)
% hObject    handle to seqn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

handles.data_dir = './ActivityDataset';
handles.anno_dir = fullfile(handles.data_dir, 'annotations');
handles.seq_str = 'seq%2.2d';
handles.anno_str = 'anno%2.2d.mat';
handles.im_str = 'frame%4.4d.jpg';
handles.interact_labels = {'no-interaction'...
    'approaching',...
    'leaving',...
    'passing-by',...
    'facing-each-other',...
    'walking-together',...
    'standing-in-a-row',...
    'standing-side-by-side',...
    'undefined'
    };

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
handles.seqn = str2double(get(hObject, 'String'));
handles.im_dir = fullfile(handles.data_dir, sprintf(handles.seq_str, handles.seqn));
handles.anno_file = fullfile(handles.anno_dir, sprintf(handles.anno_str, handles.seqn));
handles.framen = 1;
guidata(hObject, handles)
display(handles.im_dir)
display(handles.anno_file)


% --- Executes on button press in start.
function start_Callback(hObject, eventdata, handles)
% hObject    handle to start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

anno = load(handles.anno_file);
anno = anno.anno;
time_vec = 1:anno.nframe;
n_inter = 0;
axes(handles.video);
interaction = zeros([numel(anno.people), numel(anno.people), anno.nframe]);
handles.current = [];
guidata(hObject, handles)
try
    idx = int64(handles.seqn);
catch
    idx = str2double(get(handles.seqn, 'String'));
end

flag = true;
if ~flag
    saved_inter = load(fullfile(handles.anno_dir, sprintf('int%2.2d', idx)));
    saved_inter = saved_inter.interaction;
end

for m = 1:numel(anno.people)
    for n = 1:numel(anno.people)
        cla
        pause(0.3)
        if m == n
            continue;
        elseif n>=m
            n_inter = n_inter + 1;
        end
        for t = 1:10:anno.nframe
            anno.nframe - t
            ped_m = anno.people(m);
            ped_n = anno.people(n);
            if any(ped_n.time == t) && any(ped_m.time == t)
                frame = imread(fullfile(handles.im_dir, sprintf(handles.im_str, t)));
                imagesc(frame), axis image, axis off;
                rectangle('Position', ped_m.sbbs(:,ped_m.time == t), 'LineWidth', 2, 'EdgeColor','b')
                rectangle('Position', ped_n.sbbs(:,ped_n.time == t), 'LineWidth', 2, 'EdgeColor','r')
                label = anno.interaction(n_inter, t);
                handles.current.m = m;
                handles.current.n = n;
                handles.current.t = t;
                if flag
                    handles.current.label = label;
                else
                    handles.current.label = saved_inter(m, n, t);
                end
                guidata(hObject, handles)
                set(handles.label, 'String', [handles.interact_labels{label}, ' ', num2str(label)])
                set(handles.currlabel, 'String', num2str(label))
                uiwait()
                handles.current.interaction = str2double(get(handles.currlabel, 'String'));
                interaction(m, n, max(1, t-9):t) = handles.current.interaction;
            end
        end
        try
            idx = int64(handles.seqn);
        catch
            idx = str2double(get(handles.seqn, 'String'));
        end
        save(fullfile(handles.anno_dir, sprintf('int%2.2d', idx)), 'interaction')
    end
    save(fullfile(handles.anno_dir, sprintf('int%2.2d', idx)), 'interaction')
end
save(fullfile(handles.anno_dir, sprintf('int%2.2d', idx)), 'interaction')
guidata(hObject, handles)


% --- Executes on button press in nan.
function nan_Callback(hObject, eventdata, handles)
% hObject    handle to nan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.current.interaction = 9;
set(handles.currlabel, 'String', num2str(handles.current.interaction))
guidata(hObject, handles);
uiresume();
guidata(hObject, handles);



function currlabel_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to currlabel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


function pairwise_WindowKeyPressFcn(hObject, eventdata, handles)
 % determine the key that was pressed 
 keyPressed = eventdata.Key;
 switch keyPressed
     case '1'
         uicontrol(handles.nointer);
         pushbutton1_Callback(handles.nointer,[],handles);
     case '2'
         uicontrol(handles.approach);
         pushbutton1_Callback(handles.approach,[],handles);
     case '3'
         uicontrol(handles.leaving);
         pushbutton1_Callback(handles.leaving,[],handles);
     case '4'
         uicontrol(handles.passing);
         pushbutton1_Callback(handles.passing,[],handles);
     case '5'
         uicontrol(handles.facing);
         pushbutton1_Callback(handles.facing,[],handles);
     case '6'
         uicontrol(handles.walking);
         pushbutton1_Callback(handles.walking,[],handles);
     case '7'
         uicontrol(handles.row);
         pushbutton1_Callback(handles.row,[],handles);
     case '8'
         uicontrol(handles.side);
         pushbutton1_Callback(handles.side,[],handles);
     case '9'
         uicontrol(handles.nan);
         pushbutton1_Callback(handles.nan,[],handles);
     case '0'
         uicontrol(handles.same);
         pushbutton1_Callback(handles.same,[],handles);
 end

